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
	ort "github.com/yalue/onnxruntime_go"
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
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))

	if libPath := os.Getenv("ONNXRUNTIME_LIB"); libPath != "" {
		ort.SetSharedLibraryPath(libPath)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		logger.Error("failed to initialize onnxruntime", "error", err)
		os.Exit(1)
	}
	defer ort.DestroyEnvironment()

	artifactsDir := envOr("ARTIFACTS_DIR", "../artifacts")
	wp, err := tokenizer.Load(artifactsDir+"/"+contract.VocabFile, contract.DoLowerCase)
	if err != nil {
		logger.Error("failed to load vocab", "error", err)
		os.Exit(1)
	}

	fp32Sess, err := ort.NewDynamicAdvancedSession(
		artifactsDir+"/model_fp32.onnx",
		[]string{"input_ids", "attention_mask"},
		[]string{"logits"},
		nil,
	)
	if err != nil {
		logger.Error("failed to load fp32 onnx", "error", err)
		os.Exit(1)
	}
	defer fp32Sess.Destroy()

	int8Sess, err := ort.NewDynamicAdvancedSession(
		artifactsDir+"/model_int8.onnx",
		[]string{"input_ids", "attention_mask"},
		[]string{"logits"},
		nil,
	)
	if err != nil {
		logger.Error("failed to load int8 onnx", "error", err)
		os.Exit(1)
	}
	defer int8Sess.Destroy()

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
		sess := int8Sess
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

		idTensor, err := ort.NewTensor(shape, ids)
		if err != nil {
			http.Error(w, `{"error":"tensor allocation failed"}`, http.StatusInternalServerError)
			return
		}
		defer idTensor.Destroy()

		maskTensor, err := ort.NewTensor(shape, mask)
		if err != nil {
			http.Error(w, `{"error":"tensor allocation failed"}`, http.StatusInternalServerError)
			return
		}
		defer maskTensor.Destroy()

		outputs := []ort.Value{nil}
		if err := sess.Run([]ort.Value{idTensor, maskTensor}, outputs); err != nil {
			logger.Error("onnx inference failed", "error", err)
			http.Error(w, `{"error":"inference failed"}`, http.StatusInternalServerError)
			return
		}
		defer outputs[0].Destroy()

		logits := outputs[0].(*ort.Tensor[float32]).GetData()
		probs := softmax(logits)

		bestIdx := 0
		probMap := make(map[string]float32, len(contract.Labels))
		for i, lbl := range contract.Labels {
			probMap[lbl] = probs[i]
			if probs[i] > probs[bestIdx] {
				bestIdx = i
			}
		}

		resp := PredictResponse{
			Label:         contract.Labels[bestIdx],
			Confidence:    probs[bestIdx],
			Probabilities: probMap,
			Precision:     precision,
			SchemaVersion: contract.SchemaVersion,
			SchemaHash:    contract.SchemaHash,
			LatencyUs:     time.Since(start).Microseconds(),
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	})

	addr := ":" + envOr("PORT", "8080")
	srv := &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go func() {
		logger.Info("starting transformer inference server",
			"addr", addr,
			"model_id", contract.ModelID,
			"schema_hash", contract.SchemaHash,
		)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("server error", "error", err)
			os.Exit(1)
		}
	}()

	<-ctx.Done()
	logger.Info("shutting down server")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = srv.Shutdown(shutdownCtx)
}
