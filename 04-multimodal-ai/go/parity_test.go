package main_test

import (
	"encoding/base64"
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/gunawanwijaya/apprentice/04-multimodal-ai/go/internal/audio"
	"github.com/gunawanwijaya/apprentice/04-multimodal-ai/go/internal/contract"
	"github.com/gunawanwijaya/apprentice/04-multimodal-ai/go/internal/vision"
	ort "github.com/yalue/onnxruntime_go"
)

type VisionCase struct {
	PNGB64         string    `json:"png_b64"`
	ExpectedLogits []float64 `json:"expected_logits"`
	ExpectedLabel  string    `json:"expected_label"`
}

type AudioCase struct {
	WAVB64         string    `json:"wav_b64"`
	ExpectedLogits []float64 `json:"expected_logits"`
	ExpectedLabel  string    `json:"expected_label"`
}

type GoldenFile struct {
	SchemaVersion string       `json:"schema_version"`
	SchemaHash    string       `json:"schema_hash"`
	EKYCCases     []VisionCase `json:"ekyc_cases"`
	UnboxingCases []VisionCase `json:"unboxing_cases"`
	AudioCases    []AudioCase  `json:"audio_cases"`
}

func argmax(v []float32) int {
	best := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}

func runSingleInput(t *testing.T, sess *ort.DynamicAdvancedSession, shape ort.Shape, data []float32) []float32 {
	t.Helper()
	inTensor, err := ort.NewTensor(shape, data)
	if err != nil {
		t.Fatalf("new tensor: %v", err)
	}
	defer inTensor.Destroy()

	outputs := []ort.Value{nil}
	if err := sess.Run([]ort.Value{inTensor}, outputs); err != nil {
		t.Fatalf("onnx run: %v", err)
	}
	defer outputs[0].Destroy()

	raw := outputs[0].(*ort.Tensor[float32]).GetData()
	out := make([]float32, len(raw))
	copy(out, raw)
	return out
}

func TestMultimodalParity(t *testing.T) {
	if libPath := os.Getenv("ONNXRUNTIME_LIB"); libPath != "" {
		ort.SetSharedLibraryPath(libPath)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		t.Fatalf("init onnxruntime: %v", err)
	}
	defer ort.DestroyEnvironment()

	raw, err := os.ReadFile("../artifacts/golden_fixtures.json")
	if err != nil {
		t.Fatalf("read golden_fixtures.json: %v", err)
	}
	var golden GoldenFile
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatalf("unmarshal golden_fixtures.json: %v", err)
	}

	if golden.SchemaHash != contract.SchemaHash {
		t.Fatalf("schema drift! python=%s go=%s (re-run make gen)", golden.SchemaHash, contract.SchemaHash)
	}

	imgShape := ort.NewShape(1, contract.VisionChannels, contract.VisionHeight, contract.VisionWidth)
	audShape := ort.NewShape(1, contract.AudioNumFrames, 2)

	t.Run("ekyc_vision", func(t *testing.T) {
		sess, err := ort.NewDynamicAdvancedSession("../artifacts/ekyc_vision.onnx", []string{"image"}, []string{"logits"}, nil)
		if err != nil {
			t.Fatalf("load ekyc_vision.onnx: %v", err)
		}
		defer sess.Destroy()

		for i, tc := range golden.EKYCCases {
			pngBytes, err := base64.StdEncoding.DecodeString(tc.PNGB64)
			if err != nil {
				t.Fatalf("case %d b64: %v", i, err)
			}
			tensor, err := vision.DecodeToCHW(pngBytes)
			if err != nil {
				t.Fatalf("case %d preprocess: %v", i, err)
			}
			got := runSingleInput(t, sess, imgShape, tensor)
			for c := range tc.ExpectedLogits {
				if diff := math.Abs(float64(got[c]) - tc.ExpectedLogits[c]); diff > 1e-4 {
					t.Fatalf("case %d logit[%d] skew: go=%.6f py=%.6f diff=%.6f", i, c, got[c], tc.ExpectedLogits[c], diff)
				}
			}
			if lbl := contract.EKYCLabels[argmax(got)]; lbl != tc.ExpectedLabel {
				t.Fatalf("case %d label mismatch: go=%s py=%s", i, lbl, tc.ExpectedLabel)
			}
		}
	})

	t.Run("unboxing_vision", func(t *testing.T) {
		sess, err := ort.NewDynamicAdvancedSession("../artifacts/unboxing_vision.onnx", []string{"image"}, []string{"logits"}, nil)
		if err != nil {
			t.Fatalf("load unboxing_vision.onnx: %v", err)
		}
		defer sess.Destroy()

		for i, tc := range golden.UnboxingCases {
			pngBytes, err := base64.StdEncoding.DecodeString(tc.PNGB64)
			if err != nil {
				t.Fatalf("case %d b64: %v", i, err)
			}
			tensor, err := vision.DecodeToCHW(pngBytes)
			if err != nil {
				t.Fatalf("case %d preprocess: %v", i, err)
			}
			got := runSingleInput(t, sess, imgShape, tensor)
			for c := range tc.ExpectedLogits {
				if diff := math.Abs(float64(got[c]) - tc.ExpectedLogits[c]); diff > 1e-4 {
					t.Fatalf("case %d logit[%d] skew: go=%.6f py=%.6f diff=%.6f", i, c, got[c], tc.ExpectedLogits[c], diff)
				}
			}
			if lbl := contract.UnboxingLabels[argmax(got)]; lbl != tc.ExpectedLabel {
				t.Fatalf("case %d label mismatch: go=%s py=%s", i, lbl, tc.ExpectedLabel)
			}
		}
	})

	t.Run("audio_vad", func(t *testing.T) {
		sess, err := ort.NewDynamicAdvancedSession("../artifacts/audio_vad.onnx", []string{"audio_features"}, []string{"logits"}, nil)
		if err != nil {
			t.Fatalf("load audio_vad.onnx: %v", err)
		}
		defer sess.Destroy()

		for i, tc := range golden.AudioCases {
			wavBytes, err := base64.StdEncoding.DecodeString(tc.WAVB64)
			if err != nil {
				t.Fatalf("case %d b64: %v", i, err)
			}
			feats, err := audio.ExtractFeatures(wavBytes)
			if err != nil {
				t.Fatalf("case %d preprocess: %v", i, err)
			}
			got := runSingleInput(t, sess, audShape, feats)
			for c := range tc.ExpectedLogits {
				if diff := math.Abs(float64(got[c]) - tc.ExpectedLogits[c]); diff > 1e-4 {
					t.Fatalf("case %d logit[%d] skew: go=%.6f py=%.6f diff=%.6f", i, c, got[c], tc.ExpectedLogits[c], diff)
				}
			}
			if lbl := contract.AudioLabels[argmax(got)]; lbl != tc.ExpectedLabel {
				t.Fatalf("case %d label mismatch: go=%s py=%s", i, lbl, tc.ExpectedLabel)
			}
		}
	})
}
