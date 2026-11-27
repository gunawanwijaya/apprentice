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
	"github.com/gunawanwijaya/apprentice/infra/telemetry"
	ort "github.com/yalue/onnxruntime_go"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
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
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if shutdownOTel, err := telemetry.Init(ctx, "04-multimodal-ai"); err == nil {
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
	ekycSess, err := ort.NewDynamicAdvancedSession(artifactsDir+"/ekyc_vision.onnx", []string{"image"}, []string{"logits"}, nil)
	if err != nil {
		slog.Error("load ekyc_vision.onnx failed", "error", err)
		os.Exit(1)
	}
	defer ekycSess.Destroy()

	unbSess, err := ort.NewDynamicAdvancedSession(artifactsDir+"/unboxing_vision.onnx", []string{"image"}, []string{"logits"}, nil)
	if err != nil {
		slog.Error("load unboxing_vision.onnx failed", "error", err)
		os.Exit(1)
	}
	defer unbSess.Destroy()

	audSess, err := ort.NewDynamicAdvancedSession(artifactsDir+"/audio_vad.onnx", []string{"audio_features"}, []string{"logits"}, nil)
	if err != nil {
		slog.Error("load audio_vad.onnx failed", "error", err)
		os.Exit(1)
	}
	defer audSess.Destroy()

	imgShape := ort.NewShape(1, contract.VisionChannels, contract.VisionHeight, contract.VisionWidth)
	audShape := ort.NewShape(1, contract.AudioNumFrames, 2)

	meter := otel.Meter("04-multimodal-ai")
	stageDur, _ := meter.Float64Histogram(
		"multimodal.stage.duration",
		metric.WithUnit("s"),
		metric.WithDescription("Multimodal pipeline stage duration (ffmpeg demux vs ONNX)"),
		metric.WithExplicitBucketBoundaries(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
	)
	decCounter, _ := meter.Int64Counter("multimodal.decisions", metric.WithDescription("Decisions by modality and verdict"))
	dedupCounter, _ := meter.Int64Counter("multimodal.frames_dedup_saved", metric.WithDescription("Static video frames skipped via cosine dedup"))

	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"status":         "ok",
			"schema_version": contract.SchemaVersion,
			"schema_hash":    contract.SchemaHash,
		})
	})

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
		stageDur.Record(r.Context(), time.Since(start).Seconds(), metric.WithAttributes(attribute.String("stage", "vision_"+mode)))
		decCounter.Add(r.Context(), 1, metric.WithAttributes(attribute.String("modality", "image_"+mode), attribute.String("verdict", lbl)))

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(ClassifyResponse{
			Label:         lbl,
			Confidence:    conf,
			Probabilities: probMap,
			SchemaHash:    contract.SchemaHash,
			LatencyUs:     time.Since(start).Microseconds(),
		})
	})

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
		stageDur.Record(r.Context(), time.Since(start).Seconds(), metric.WithAttributes(attribute.String("stage", "audio_vad")))
		decCounter.Add(r.Context(), 1, metric.WithAttributes(attribute.String("modality", "audio_vad"), attribute.String("verdict", lbl)))

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(ClassifyResponse{
			Label:         lbl,
			Confidence:    conf,
			Probabilities: probMap,
			SchemaHash:    contract.SchemaHash,
			LatencyUs:     time.Since(start).Microseconds(),
		})
	})

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
				http.Error(w, `{"error":"video demux failed"}`, http.StatusBadRequest)
				return
			}
			demuxDur := time.Since(demuxStart)

			infStart := time.Now()
			var (
				framePreds   []FramePrediction
				dedupCount   int
				audioVerdict = "no_audio_track"
				audioConf    float32
			)

			g, _ := errgroup.WithContext(r.Context())
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
					framePreds = append(framePreds, FramePrediction{FrameIndex: idx, Label: lbl, Confidence: conf})
				}
				return nil
			})
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
				http.Error(w, `{"error":"multimodal inference failed"}`, http.StatusInternalServerError)
				return
			}
			infDur := time.Since(infStart)

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

			stageDur.Record(r.Context(), demuxDur.Seconds(), metric.WithAttributes(attribute.String("stage", "ffmpeg_demux")))
			stageDur.Record(r.Context(), infDur.Seconds(), metric.WithAttributes(attribute.String("stage", "concurrent_onnx")))
			decCounter.Add(r.Context(), 1, metric.WithAttributes(attribute.String("modality", "video_"+useCase), attribute.String("verdict", verdict)))
			if skipped := len(demuxed.Frames) - dedupCount; skipped > 0 {
				dedupCounter.Add(r.Context(), int64(skipped))
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
				DemuxUs:           demuxDur.Microseconds(),
				InferenceUs:       infDur.Microseconds(),
				TotalLatencyUs:    time.Since(totalStart).Microseconds(),
			})
		}
	}

	mux.HandleFunc("POST /ekyc/video", handleVideo("ekyc", ekycSess, contract.EKYCLabels))
	mux.HandleFunc("POST /unboxing/video", handleVideo("unboxing", unbSess, contract.UnboxingLabels))

	addr := ":" + envOr("PORT", "8084")
	srv := &http.Server{
		Addr:              addr,
		Handler:           telemetry.Middleware("04-multimodal-ai", mux),
		ReadHeaderTimeout: 5 * time.Second,
	}

	go func() {
		slog.Info("04-multimodal-ai listening", "addr", addr, "schema_hash", contract.SchemaHash)
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
