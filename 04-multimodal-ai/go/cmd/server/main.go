package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"math"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/gunawanwijaya/apprentice/04-multimodal-ai/go/internal/audio"
	"github.com/gunawanwijaya/apprentice/04-multimodal-ai/go/internal/contract"
	"github.com/gunawanwijaya/apprentice/04-multimodal-ai/go/internal/video"
	"github.com/gunawanwijaya/apprentice/04-multimodal-ai/go/internal/vision"
	ort "github.com/yalue/onnxruntime_go"
	"golang.org/x/sync/errgroup"
)

type ClassifyResponse struct {
	Label         string             `json:"label"`
	Confidence    float32            `json:"confidence"`
	Probabilities map[string]float32 `json:"probabilities"`
	SchemaHash    string             `json:"schema_hash"`
	LatencyUs     int64              `json:"latency_us"`
}

type FramePrediction struct {
	FrameIndex int     `json:"frame_index"`
	Label      string  `json:"label"`
	Confidence float32 `json:"confidence"`
}

type VideoAnalysisResponse struct {
	UseCase           string            `json:"use_case"`
	Verdict           string            `json:"verdict"`
	FramesExtracted   int               `json:"frames_extracted"`
	FramesAfterDedup  int               `json:"frames_after_dedup"`
	DominantVisual    string            `json:"dominant_visual"`
	VisualPredictions []FramePrediction `json:"visual_predictions"`
	AudioVerdict      string            `json:"audio_verdict"`
	AudioConfidence   float32           `json:"audio_confidence"`
	SchemaHash        string            `json:"schema_hash"`
	DemuxUs           int64             `json:"demux_us"`
	InferenceUs       int64             `json:"inference_us"`
	TotalLatencyUs    int64             `json:"total_latency_us"`
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

func runONNX(sess *ort.DynamicAdvancedSession, shape ort.Shape, data []float32) ([]float32, error) {
	inTensor, err := ort.NewTensor(shape, data)
	if err != nil {
		return nil, err
	}
	defer inTensor.Destroy()

	outputs := []ort.Value{nil}
	if err := sess.Run([]ort.Value{inTensor}, outputs); err != nil {
		return nil, err
	}
	defer outputs[0].Destroy()

	raw := outputs[0].(*ort.Tensor[float32]).GetData()
	out := make([]float32, len(raw))
	copy(out, raw)
	return out, nil
}

func topClass(probs []float32, labels []string) (string, float32, map[string]float32) {
	best := 0
	probMap := make(map[string]float32, len(labels))
	for i, lbl := range labels {
		probMap[lbl] = probs[i]
		if probs[i] > probs[best] {
			best = i
		}
	}
	return labels[best], probs[best], probMap
}

// readUpload accepts either multipart/form-data ("file" field) or raw binary body.
func readUpload(w http.ResponseWriter, r *http.Request, maxBytes int64) ([]byte, error) {
	r.Body = http.MaxBytesReader(w, r.Body, maxBytes)
	ct := r.Header.Get("Content-Type")
	if strings.HasPrefix(ct, "multipart/form-data") {
		if err := r.ParseMultipartForm(maxBytes); err != nil {
			return nil, err
		}
		f, _, err := r.FormFile("file")
		if err != nil {
			return nil, err
		}
		defer f.Close()
		return io.ReadAll(f)
	}
	return io.ReadAll(r.Body)
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
	ekycSess, err := ort.NewDynamicAdvancedSession(artifactsDir+"/ekyc_vision.onnx", []string{"image"}, []string{"logits"}, nil)
	if err != nil {
		logger.Error("load ekyc_vision.onnx failed", "error", err)
		os.Exit(1)
	}
	defer ekycSess.Destroy()

	unbSess, err := ort.NewDynamicAdvancedSession(artifactsDir+"/unboxing_vision.onnx", []string{"image"}, []string{"logits"}, nil)
	if err != nil {
		logger.Error("load unboxing_vision.onnx failed", "error", err)
		os.Exit(1)
	}
	defer unbSess.Destroy()

	audSess, err := ort.NewDynamicAdvancedSession(artifactsDir+"/audio_vad.onnx", []string{"audio_features"}, []string{"logits"}, nil)
	if err != nil {
		logger.Error("load audio_vad.onnx failed", "error", err)
		os.Exit(1)
	}
	defer audSess.Destroy()

	imgShape := ort.NewShape(1, contract.VisionChannels, contract.VisionHeight, contract.VisionWidth)
	audShape := ort.NewShape(1, contract.AudioNumFrames, 2)

	mux := http.NewServeMux()

	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"status":         "ok",
			"schema_version": contract.SchemaVersion,
			"schema_hash":    contract.SchemaHash,
		})
	})

	// Atomic Vision Endpoint: POST /image/classify?mode=ekyc|unboxing
	mux.HandleFunc("POST /image/classify", func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		defer r.Body.Close()

		mode := r.URL.Query().Get("mode")
		if mode == "" {
			mode = "ekyc"
		}
		var sess *ort.DynamicAdvancedSession
		var labels []string
		switch mode {
		case "ekyc":
			sess, labels = ekycSess, contract.EKYCLabels
		case "unboxing":
			sess, labels = unbSess, contract.UnboxingLabels
		default:
			http.Error(w, `{"error":"mode must be 'ekyc' or 'unboxing'"}`, http.StatusBadRequest)
			return
		}

		data, err := readUpload(w, r, 5<<20)
		if err != nil || len(data) == 0 {
			http.Error(w, `{"error":"failed to read image payload"}`, http.StatusBadRequest)
			return
		}
		tensor, err := vision.DecodeToCHW(data)
		if err != nil {
			http.Error(w, `{"error":"invalid PNG/JPEG image"}`, http.StatusBadRequest)
			return
		}
		logits, err := runONNX(sess, imgShape, tensor)
		if err != nil {
			http.Error(w, `{"error":"vision inference failed"}`, http.StatusInternalServerError)
			return
		}
		lbl, conf, probMap := topClass(softmax(logits), labels)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(ClassifyResponse{
			Label:         lbl,
			Confidence:    conf,
			Probabilities: probMap,
			SchemaHash:    contract.SchemaHash,
			LatencyUs:     time.Since(start).Microseconds(),
		})
	})

	// Atomic Audio Endpoint: POST /audio/vad
	mux.HandleFunc("POST /audio/vad", func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		defer r.Body.Close()

		data, err := readUpload(w, r, 5<<20)
		if err != nil || len(data) == 0 {
			http.Error(w, `{"error":"failed to read WAV payload"}`, http.StatusBadRequest)
			return
		}
		feats, err := audio.ExtractFeatures(data)
		if err != nil {
			http.Error(w, `{"error":"invalid 16kHz mono PCM WAV"}`, http.StatusBadRequest)
			return
		}
		logits, err := runONNX(audSess, audShape, feats)
		if err != nil {
			http.Error(w, `{"error":"audio inference failed"}`, http.StatusInternalServerError)
			return
		}
		lbl, conf, probMap := topClass(softmax(logits), contract.AudioLabels)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(ClassifyResponse{
			Label:         lbl,
			Confidence:    conf,
			Probabilities: probMap,
			SchemaHash:    contract.SchemaHash,
			LatencyUs:     time.Since(start).Microseconds(),
		})
	})

	// Shared Video -> (Vision || Audio) Concurrent Handler
	handleVideo := func(useCase string, visSess *ort.DynamicAdvancedSession, visLabels []string) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			totalStart := time.Now()
			defer r.Body.Close()

			videoBytes, err := readUpload(w, r, 15<<20)
			if err != nil || len(videoBytes) == 0 {
				http.Error(w, `{"error":"failed to read video payload"}`, http.StatusBadRequest)
				return
			}

			demuxStart := time.Now()
			demuxed, err := video.Demux(r.Context(), videoBytes)
			if err != nil {
				logger.Error("video demux failed", "error", err)
				http.Error(w, `{"error":"video demux failed (ensure valid mp4/webm)"}`, http.StatusBadRequest)
				return
			}
			demuxUs := time.Since(demuxStart).Microseconds()

			infStart := time.Now()
			var (
				framePreds   []FramePrediction
				dedupCount   int
				audioVerdict = "no_audio_track"
				audioConf    float32
			)

			g, _ := errgroup.WithContext(r.Context())

			// Goroutine 1: Vision (Frame Deduplication -> 2D-CNN ONNX)
			g.Go(func() error {
				var lastTensor []float32
				for idx, framePNG := range demuxed.Frames {
					t, err := vision.DecodeToCHW(framePNG)
					if err != nil {
						return err
					}
					if lastTensor != nil && vision.CosineSimilarity(lastTensor, t) >= contract.VideoDedupThreshold {
						continue
					}
					lastTensor = t
					dedupCount++

					logits, err := runONNX(visSess, imgShape, t)
					if err != nil {
						return err
					}
					lbl, conf, _ := topClass(softmax(logits), visLabels)
					framePreds = append(framePreds, FramePrediction{
						FrameIndex: idx,
						Label:      lbl,
						Confidence: conf,
					})
				}
				return nil
			})

			// Goroutine 2: Audio (16kHz PCM -> 1D-CNN VAD ONNX)
			g.Go(func() error {
				if len(demuxed.AudioWAV) == 0 {
					return nil
				}
				feats, err := audio.ExtractFeatures(demuxed.AudioWAV)
				if err != nil {
					return err
				}
				logits, err := runONNX(audSess, audShape, feats)
				if err != nil {
					return err
				}
				audioVerdict, audioConf, _ = topClass(softmax(logits), contract.AudioLabels)
				return nil
			})

			if err := g.Wait(); err != nil {
				logger.Error("multimodal inference failed", "error", err)
				http.Error(w, `{"error":"multimodal inference failed"}`, http.StatusInternalServerError)
				return
			}
			infUs := time.Since(infStart).Microseconds()

			// Majority vote across deduplicated frames
			counts := make(map[string]int)
			dominant := "unknown"
			bestCount := 0
			for _, fp := range framePreds {
				counts[fp.Label]++
				if counts[fp.Label] > bestCount {
					bestCount = counts[fp.Label]
					dominant = fp.Label
				}
			}

			verdict := "manual_review"
			if useCase == "ekyc" {
				if dominant == "clean_document" && audioVerdict == "speech_present" {
					verdict = "pass_ekyc_gate"
				} else if dominant != "clean_document" {
					verdict = "reject_visual_" + dominant
				} else {
					verdict = "reject_audio_" + audioVerdict
				}
			} else {
				if dominant == "damaged_box" || dominant == "empty_box" {
					verdict = "escalate_claim_" + dominant
				} else if dominant == "intact_parcel" {
					verdict = "auto_close_intact"
				} else {
					verdict = "request_resubmission_" + dominant
				}
			}

			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(VideoAnalysisResponse{
				UseCase:           useCase,
				Verdict:           verdict,
				FramesExtracted:   len(demuxed.Frames),
				FramesAfterDedup:  dedupCount,
				DominantVisual:    dominant,
				VisualPredictions: framePreds,
				AudioVerdict:      audioVerdict,
				AudioConfidence:   audioConf,
				SchemaHash:        contract.SchemaHash,
				DemuxUs:           demuxUs,
				InferenceUs:       infUs,
				TotalLatencyUs:    time.Since(totalStart).Microseconds(),
			})
		}
	}

	mux.HandleFunc("POST /ekyc/video", handleVideo("ekyc", ekycSess, contract.EKYCLabels))
	mux.HandleFunc("POST /unboxing/video", handleVideo("unboxing", unbSess, contract.UnboxingLabels))

	addr := ":" + envOr("PORT", "8080")
	srv := &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go func() {
		logger.Info("starting multimodal inference server",
			"addr", addr,
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
