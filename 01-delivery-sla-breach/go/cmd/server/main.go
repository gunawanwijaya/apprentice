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

	"github.com/gunawanwijaya/apprentice/01-delivery-sla-breach/go/internal/contract"
	ort "github.com/yalue/onnxruntime_go"
)

type CQRCalibration struct {
	CoverageTarget float64 `json:"coverage_target"`
	CQRMarginDays  float64 `json:"cqr_margin_days"`
}

type PredictResponse struct {
	IsLate                bool    `json:"is_late"`
	Probability           float32 `json:"probability"`
	PredictedDeliveryDays float64 `json:"predicted_delivery_days"`
	ETAMinDays            float64 `json:"eta_min_days"`
	ETAMaxDays            float64 `json:"eta_max_days"`
	ETACoverageTarget     float64 `json:"eta_coverage_target"`
	Threshold             float32 `json:"threshold"`
	SchemaVersion         string  `json:"schema_version"`
	SchemaHash            string  `json:"schema_hash"`
	LatencyUs             int64   `json:"latency_us"`
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func runScalarONNX(sess *ort.DynamicAdvancedSession, inTensor *ort.Tensor[float32]) (float64, error) {
	out := []ort.Value{nil}
	if err := sess.Run([]ort.Value{inTensor}, out); err != nil {
		return 0, err
	}
	defer out[0].Destroy()
	return float64(out[0].(*ort.Tensor[float32]).GetData()[0]), nil
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

	calRaw, err := os.ReadFile(artifactsDir + "/cqr_calibration.json")
	if err != nil {
		logger.Error("failed to read cqr_calibration.json", "error", err)
		os.Exit(1)
	}
	var cal CQRCalibration
	if err := json.Unmarshal(calRaw, &cal); err != nil {
		logger.Error("failed to parse cqr_calibration.json", "error", err)
		os.Exit(1)
	}

	clfSess, err := ort.NewDynamicAdvancedSession(artifactsDir+"/model.onnx", []string{"input"}, []string{"label", "probabilities"}, nil)
	if err != nil {
		logger.Error("failed to load model.onnx", "error", err)
		os.Exit(1)
	}
	defer clfSess.Destroy()

	p10Sess, err := ort.NewDynamicAdvancedSession(artifactsDir+"/eta_p10.onnx", []string{"input"}, []string{"variable"}, nil)
	if err != nil {
		logger.Error("failed to load eta_p10.onnx", "error", err)
		os.Exit(1)
	}
	defer p10Sess.Destroy()

	p50Sess, err := ort.NewDynamicAdvancedSession(artifactsDir+"/eta_p50.onnx", []string{"input"}, []string{"variable"}, nil)
	if err != nil {
		logger.Error("failed to load eta_p50.onnx", "error", err)
		os.Exit(1)
	}
	defer p50Sess.Destroy()

	p90Sess, err := ort.NewDynamicAdvancedSession(artifactsDir+"/eta_p90.onnx", []string{"input"}, []string{"variable"}, nil)
	if err != nil {
		logger.Error("failed to load eta_p90.onnx", "error", err)
		os.Exit(1)
	}
	defer p90Sess.Destroy()

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

		clfOut := []ort.Value{nil, nil}
		if err := clfSess.Run([]ort.Value{inTensor}, clfOut); err != nil {
			logger.Error("classifier inference failed", "error", err)
			http.Error(w, `{"error":"inference failed"}`, http.StatusInternalServerError)
			return
		}
		prob := clfOut[1].(*ort.Tensor[float32]).GetData()[1]
		clfOut[0].Destroy()
		clfOut[1].Destroy()

		rawP10, err := runScalarONNX(p10Sess, inTensor)
		if err != nil {
			logger.Error("p10 inference failed", "error", err)
			http.Error(w, `{"error":"inference failed"}`, http.StatusInternalServerError)
			return
		}
		rawP50, err := runScalarONNX(p50Sess, inTensor)
		if err != nil {
			logger.Error("p50 inference failed", "error", err)
			http.Error(w, `{"error":"inference failed"}`, http.StatusInternalServerError)
			return
		}
		rawP90, err := runScalarONNX(p90Sess, inTensor)
		if err != nil {
			logger.Error("p90 inference failed", "error", err)
			http.Error(w, `{"error":"inference failed"}`, http.StatusInternalServerError)
			return
		}

		etaMin := math.Max(1.0, rawP10-cal.CQRMarginDays)
		etaMax := math.Max(etaMin, rawP90+cal.CQRMarginDays)
		etaP50 := math.Min(math.Max(rawP50, etaMin), etaMax)

		resp := PredictResponse{
			IsLate:                prob >= contract.Threshold,
			Probability:           prob,
			PredictedDeliveryDays: math.Round(etaP50*10) / 10,
			ETAMinDays:            math.Round(etaMin*10) / 10,
			ETAMaxDays:            math.Round(etaMax*10) / 10,
			ETACoverageTarget:     cal.CoverageTarget,
			Threshold:             contract.Threshold,
			SchemaVersion:         contract.SchemaVersion,
			SchemaHash:            contract.SchemaHash,
			LatencyUs:             time.Since(start).Microseconds(),
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
			"schema_version", contract.SchemaVersion,
			"schema_hash", contract.SchemaHash,
			"cqr_margin_days", cal.CQRMarginDays,
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
