package main

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/gunawanwijaya/apprentice/02-classic-nlp/go/internal/bm25"
	"github.com/gunawanwijaya/apprentice/02-classic-nlp/go/internal/contract"
	"github.com/gunawanwijaya/apprentice/02-classic-nlp/go/internal/pii"
	"github.com/gunawanwijaya/apprentice/02-classic-nlp/go/internal/viterbi"
	ort "github.com/yalue/onnxruntime_go"
)

type TextRequest struct {
	Text string `json:"text"`
}

type SearchRequest struct {
	Query string `json:"query"`
	TopK  int    `json:"top_k,omitempty"`
}

type NERRequest struct {
	Text   string   `json:"text,omitempty"`
	Tokens []string `json:"tokens,omitempty"`
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func decodeStrict(w http.ResponseWriter, r *http.Request, dst any) bool {
	defer r.Body.Close()
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<16))
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid payload: " + err.Error()})
		return false
	}
	return true
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func runTextONNX(sess *ort.DynamicAdvancedSession, text string) (string, float32, error) {
	inTensor, err := ort.NewStringTensor(ort.NewShape(1, 1))
	if err != nil {
		return "", 0, err
	}
	defer inTensor.Destroy()

	if err := inTensor.SetContents([]string{text}); err != nil {
		return "", 0, err
	}

	outputs := []ort.Value{nil, nil}
	if err := sess.Run([]ort.Value{inTensor}, outputs); err != nil {
		return "", 0, err
	}
	defer outputs[0].Destroy()
	defer outputs[1].Destroy()

	lbls, err := outputs[0].(*ort.StringTensor).GetContents()
	if err != nil {
		return "", 0, err
	}
	probs := outputs[1].(*ort.Tensor[float32]).GetData()
	maxP := float32(0)
	for _, p := range probs {
		if p > maxP {
			maxP = p
		}
	}
	return lbls[0], maxP, nil
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

	intentSess, err := ort.NewDynamicAdvancedSession(artifactsDir+"/intent.onnx", []string{"input"}, []string{"label", "probabilities"}, nil)
	if err != nil {
		logger.Error("failed to load intent.onnx", "error", err)
		os.Exit(1)
	}
	defer intentSess.Destroy()

	queueSess, err := ort.NewDynamicAdvancedSession(artifactsDir+"/queue.onnx", []string{"input"}, []string{"label", "probabilities"}, nil)
	if err != nil {
		logger.Error("failed to load queue.onnx", "error", err)
		os.Exit(1)
	}
	defer queueSess.Destroy()

	prioSess, err := ort.NewDynamicAdvancedSession(artifactsDir+"/priority.onnx", []string{"input"}, []string{"label", "probabilities"}, nil)
	if err != nil {
		logger.Error("failed to load priority.onnx", "error", err)
		os.Exit(1)
	}
	defer prioSess.Destroy()

	bm25Idx, err := bm25.Load(artifactsDir+"/bm25_corpus.json", contract.TokenPattern)
	if err != nil {
		logger.Error("failed to load bm25_corpus.json", "error", err)
		os.Exit(1)
	}

	nerModel, err := viterbi.Load(artifactsDir + "/ner_crf.json")
	if err != nil {
		logger.Error("failed to load ner_crf.json", "error", err)
		os.Exit(1)
	}

	mux := http.NewServeMux()

	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{
			"status":         "ok",
			"schema_version": contract.SchemaVersion,
			"schema_hash":    contract.SchemaHash,
		})
	})

	mux.HandleFunc("POST /redact", func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		var req TextRequest
		if !decodeStrict(w, r, &req) {
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"redacted":   pii.Redact(req.Text),
			"latency_us": time.Since(start).Microseconds(),
		})
	})

	mux.HandleFunc("POST /intent", func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		var req TextRequest
		if !decodeStrict(w, r, &req) {
			return
		}
		redacted := pii.Redact(req.Text)
		label, prob, err := runTextONNX(intentSess, redacted)
		if err != nil {
			logger.Error("intent inference failed", "error", err)
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "inference failed"})
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"intent":        label,
			"probability":   prob,
			"redacted_text": redacted,
			"schema_hash":   contract.SchemaHash,
			"latency_us":    time.Since(start).Microseconds(),
		})
	})

	mux.HandleFunc("POST /search", func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		var req SearchRequest
		if !decodeStrict(w, r, &req) {
			return
		}
		topK := req.TopK
		if topK <= 0 {
			topK = contract.BM25TopK
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"results":    bm25Idx.Search(req.Query, topK),
			"latency_us": time.Since(start).Microseconds(),
		})
	})

	mux.HandleFunc("POST /ner", func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		var req NERRequest
		if !decodeStrict(w, r, &req) {
			return
		}
		tokens := req.Tokens
		if len(tokens) == 0 && strings.TrimSpace(req.Text) != "" {
			tokens = strings.Fields(req.Text)
		}
		tags := nerModel.Predict(tokens)
		entities := make([]map[string]string, len(tokens))
		for i := range tokens {
			entities[i] = map[string]string{"token": tokens[i], "tag": tags[i]}
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"entities":   entities,
			"latency_us": time.Since(start).Microseconds(),
		})
	})

	mux.HandleFunc("POST /triage", func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		var req TextRequest
		if !decodeStrict(w, r, &req) {
			return
		}
		redacted := pii.Redact(req.Text)
		queue, qProb, err := runTextONNX(queueSess, redacted)
		if err != nil {
			logger.Error("queue inference failed", "error", err)
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "inference failed"})
			return
		}
		prio, pProb, err := runTextONNX(prioSess, redacted)
		if err != nil {
			logger.Error("priority inference failed", "error", err)
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "inference failed"})
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"queue":         queue,
			"queue_prob":    qProb,
			"priority":      prio,
			"priority_prob": pProb,
			"redacted_text": redacted,
			"schema_hash":   contract.SchemaHash,
			"latency_us":    time.Since(start).Microseconds(),
		})
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
		logger.Info("starting classic-nlp server", "addr", addr, "schema_version", contract.SchemaVersion, "schema_hash", contract.SchemaHash)
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
