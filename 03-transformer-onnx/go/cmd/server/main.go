package main

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"math"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gunawanwijaya/apprentice/03-transformer-onnx/go/internal/contract"
	"github.com/gunawanwijaya/apprentice/03-transformer-onnx/go/internal/tokenizer"
	"github.com/gunawanwijaya/apprentice/infra/telemetry"
	ort "github.com/yalue/onnxruntime_go"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
)

type PredictRequest struct {
	Text      string `json:"text"`
	Precision string `json:"precision,omitempty"` // "int8" (default) or "fp32"
}

type PredictResponse struct {
	Label         string             `json:"label"`
	Confidence    float32            `json:"confidence"`
	Probabilities map[string]float32 `json:"probabilities"`
	Precision     string             `json:"precision"`
	SchemaVersion string             `json:"schema_version"`
	SchemaHash    string             `json:"schema_hash"`
	LatencyUs     int64              `json:"latency_us"`
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func softmax(logits []float32) []float32 {
	maxVal := logits[0]
	for _, v := range logits[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	probs := make([]float32, len(logits))
	var sum float64
	for i, v := range logits {
		e := math.Exp(float64(v - maxVal))
		probs[i] = float32(e)
		sum += e
	}
	for i := range probs {
		probs[i] = float32(float64(probs[i]) / sum)
	}
	return probs
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if shutdownOTel, err := telemetry.Init(ctx, "03-transformer-onnx"); err == nil {
		defer shutdownOTel(context.Background())
	}

	if libPath := os.Getenv("ONNXRUNTIME_LIB"); libPath != "" {
		ort.SetSharedLibraryPath(libPath)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		slog.Error("failed to initialize onnxruntime", "error", err)
		os.Exit(1)
	}
	defer ort.DestroyEnvironment()

	artifactsDir := envOr("ARTIFACTS_DIR", "../artifacts")
	wp, err := tokenizer.Load(artifactsDir+"/"+contract.VocabFile, contract.DoLowerCase)
	if err != nil {
		slog.Error("failed to load vocab", "error", err)
		os.Exit(1)
	}

	fp32Sess, err := ort.NewDynamicAdvancedSession(artifactsDir+"/model_fp32.onnx", []string{"input_ids", "attention_mask"}, []string{"logits"}, nil)
	if err != nil {
		slog.Error("failed to load fp32 onnx", "error", err)
		os.Exit(1)
	}
	defer fp32Sess.Destroy()

	int8Sess, err := ort.NewDynamicAdvancedSession(artifactsDir+"/model_int8.onnx", []string{"input_ids", "attention_mask"}, []string{"logits"}, nil)
	if err != nil {
		slog.Error("failed to load int8 onnx", "error", err)
		os.Exit(1)
	}
	defer int8Sess.Destroy()

	meter := otel.Meter("03-transformer-onnx")
	infDur, _ := meter.Float64Histogram(
		"transformer.inference.duration",
		metric.WithUnit("s"),
		metric.WithDescription("ONNX encoder inference latency by precision (fp32 vs int8)"),
		metric.WithExplicitBucketBoundaries(0.0002, 0.0005, 0.0008, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.01, 0.025),
	)
	predCounter, _ := meter.Int64Counter("transformer.predictions", metric.WithDescription("Predictions by precision and label"))
	confHist, _ := meter.Float64Histogram(
		"transformer.confidence",
		metric.WithDescription("Top-1 softmax confidence distribution"),
		metric.WithExplicitBucketBoundaries(0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
	)

	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"status":         "ok",
			"model_id":       contract.ModelID,
			"schema_version": contract.SchemaVersion,
			"schema_hash":    contract.SchemaHash,
		})
	})

	mux.HandleFunc("POST /predict", func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		defer r.Body.Close()

		var req PredictRequest
		dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<16))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&req); err != nil || req.Text == "" {
			http.Error(w, `{"error":"invalid request payload: text is required"}`, http.StatusBadRequest)
			return
		}

		precision := req.Precision
		if precision == "" {
			precision = "int8"
		}
		var sess *ort.DynamicAdvancedSession
		switch precision {
		case "int8":
			sess = int8Sess
		case "fp32":
			sess = fp32Sess
		default:
			http.Error(w, `{"error":"precision must be 'int8' or 'fp32'"}`, http.StatusBadRequest)
			return
		}

		ids, mask := wp.Encode(req.Text, contract.MaxSeqLen)
		shape := ort.NewShape(1, int64(contract.MaxSeqLen))
		idTensor, err1 := ort.NewTensor(shape, ids)
		maskTensor, err2 := ort.NewTensor(shape, mask)
		if err1 != nil || err2 != nil {
			http.Error(w, `{"error":"tensor allocation failed"}`, http.StatusInternalServerError)
			return
		}
		defer idTensor.Destroy()
		defer maskTensor.Destroy()

		onnxStart := time.Now()
		outputs := []ort.Value{nil}
		if err := sess.Run([]ort.Value{idTensor, maskTensor}, outputs); err != nil {
			http.Error(w, `{"error":"inference failed"}`, http.StatusInternalServerError)
			return
		}
		defer outputs[0].Destroy()
		onnxSec := time.Since(onnxStart).Seconds()

		probs := softmax(outputs[0].(*ort.Tensor[float32]).GetData())
		bestIdx := 0
		probMap := make(map[string]float32, len(contract.Labels))
		for i, lbl := range contract.Labels {
			probMap[lbl] = probs[i]
			if probs[i] > probs[bestIdx] {
				bestIdx = i
			}
		}

		lbl := contract.Labels[bestIdx]
		conf := probs[bestIdx]
		infDur.Record(r.Context(), onnxSec, metric.WithAttributes(attribute.String("precision", precision)))
		predCounter.Add(r.Context(), 1, metric.WithAttributes(attribute.String("precision", precision), attribute.String("label", lbl)))
		confHist.Record(r.Context(), float64(conf), metric.WithAttributes(attribute.String("precision", precision)))

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(PredictResponse{
			Label:         lbl,
			Confidence:    conf,
			Probabilities: probMap,
			Precision:     precision,
			SchemaVersion: contract.SchemaVersion,
			SchemaHash:    contract.SchemaHash,
			LatencyUs:     time.Since(start).Microseconds(),
		})
	})

	addr := ":" + envOr("PORT", "8083")
	srv := &http.Server{
		Addr:              addr,
		Handler:           telemetry.Middleware("03-transformer-onnx", mux),
		ReadHeaderTimeout: 5 * time.Second,
	}

	go func() {
		slog.Info("03-transformer-onnx listening", "addr", addr, "model_id", contract.ModelID, "schema_hash", contract.SchemaHash)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			slog.Error("server error", "error", err)
			os.Exit(1)
		}
	}()

	<-ctx.Done()
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = srv.Shutdown(shutdownCtx)
}
