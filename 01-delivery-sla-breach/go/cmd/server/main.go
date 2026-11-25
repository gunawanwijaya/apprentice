package main

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gunawanwijaya/apprentice/01-delivery-sla-breach/go/internal/contract"
	ort "github.com/yalue/onnxruntime_go"
)

type PredictResponse struct {
	IsLate        bool    `json:"is_late"`
	Probability   float32 `json:"probability"`
	Threshold     float32 `json:"threshold"`
	SchemaVersion string  `json:"schema_version"`
	SchemaHash    string  `json:"schema_hash"`
	LatencyUs     int64   `json:"latency_us"`
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
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

	modelPath := envOr("MODEL_PATH", "../artifacts/model.onnx")
	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input"},
		[]string{"label", "probabilities"},
		nil,
	)
	if err != nil {
		logger.Error("failed to load onnx model", "path", modelPath, "error", err)
		os.Exit(1)
	}
	defer session.Destroy()

	mux := http.NewServeMux()

	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"status":         "ok",
			"schema_version": contract.SchemaVersion,
			"schema_hash":    contract.SchemaHash,
		})
	})

	mux.HandleFunc("POST /predict", func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		defer r.Body.Close()

		var req contract.PredictRequest
		dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<16))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&req); err != nil {
			http.Error(w, `{"error":"invalid request payload: `+err.Error()+`"}`, http.StatusBadRequest)
			return
		}

		inTensor, err := ort.NewTensor(ort.NewShape(1, contract.FeatureCount), req.ToFloat32Slice())
		if err != nil {
			logger.Error("input tensor allocation failed", "error", err)
			http.Error(w, `{"error":"inference setup failed"}`, http.StatusInternalServerError)
			return
		}
		defer inTensor.Destroy()

		outputs := []ort.Value{nil, nil}
		if err := session.Run([]ort.Value{inTensor}, outputs); err != nil {
			logger.Error("onnx inference failed", "error", err)
			http.Error(w, `{"error":"inference failed"}`, http.StatusInternalServerError)
			return
		}
		defer outputs[0].Destroy()
		defer outputs[1].Destroy()

		prob := outputs[1].(*ort.Tensor[float32]).GetData()[1]
		resp := PredictResponse{
			IsLate:        prob >= contract.Threshold,
			Probability:   prob,
			Threshold:     contract.Threshold,
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
		logger.Info("starting inference server",
			"addr", addr,
			"model", modelPath,
			"schema_version", contract.SchemaVersion,
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
