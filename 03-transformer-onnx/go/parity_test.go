package main_test

import (
	"encoding/json"
	"math"
	"os"
	"testing"
	"time"

	"github.com/gunawanwijaya/apprentice/03-transformer-onnx/go/internal/contract"
	"github.com/gunawanwijaya/apprentice/03-transformer-onnx/go/internal/tokenizer"
	ort "github.com/yalue/onnxruntime_go"
)

type GoldenFile struct {
	SchemaVersion string `json:"schema_version"`
	SchemaHash    string `json:"schema_hash"`
	Cases         []struct {
		Text          string    `json:"text"`
		InputIDs      []int64   `json:"input_ids"`
		AttentionMask []int64   `json:"attention_mask"`
		FP32Logits    []float64 `json:"fp32_logits"`
		INT8Logits    []float64 `json:"int8_logits"`
		ExpectedLabel string    `json:"expected_label"`
	} `json:"cases"`
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

func runSession(t *testing.T, sess *ort.DynamicAdvancedSession, ids, mask []int64) ([]float32, time.Duration) {
	t.Helper()
	shape := ort.NewShape(1, int64(contract.MaxSeqLen))
	idTensor, err := ort.NewTensor(shape, ids)
	if err != nil {
		t.Fatalf("id tensor: %v", err)
	}
	defer idTensor.Destroy()

	maskTensor, err := ort.NewTensor(shape, mask)
	if err != nil {
		t.Fatalf("mask tensor: %v", err)
	}
	defer maskTensor.Destroy()

	outputs := []ort.Value{nil}
	start := time.Now()
	if err := sess.Run([]ort.Value{idTensor, maskTensor}, outputs); err != nil {
		t.Fatalf("onnx run: %v", err)
	}
	elapsed := time.Since(start)
	defer outputs[0].Destroy()

	data := outputs[0].(*ort.Tensor[float32]).GetData()
	out := make([]float32, len(data))
	copy(out, data)
	return out, elapsed
}

func TestTransformerParity(t *testing.T) {
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

	// Gate 1: SchemaHash check
	if golden.SchemaHash != contract.SchemaHash {
		t.Fatalf("schema drift! python=%s go=%s (re-run make gen)", golden.SchemaHash, contract.SchemaHash)
	}

	wp, err := tokenizer.Load("../artifacts/"+contract.VocabFile, contract.DoLowerCase)
	if err != nil {
		t.Fatalf("load vocab: %v", err)
	}

	fp32Sess, err := ort.NewDynamicAdvancedSession("../artifacts/model_fp32.onnx", []string{"input_ids", "attention_mask"}, []string{"logits"}, nil)
	if err != nil {
		t.Fatalf("load fp32 onnx: %v", err)
	}
	defer fp32Sess.Destroy()

	int8Sess, err := ort.NewDynamicAdvancedSession("../artifacts/model_int8.onnx", []string{"input_ids", "attention_mask"}, []string{"logits"}, nil)
	if err != nil {
		t.Fatalf("load int8 onnx: %v", err)
	}
	defer int8Sess.Destroy()

	var fp32Total, int8Total time.Duration
	for i, tc := range golden.Cases {
		// Gate 2: Exact WordPiece Tokenization Parity
		gotIDs, gotMask := wp.Encode(tc.Text, contract.MaxSeqLen)
		for j := 0; j < contract.MaxSeqLen; j++ {
			if gotIDs[j] != tc.InputIDs[j] || gotMask[j] != tc.AttentionMask[j] {
				t.Fatalf("case %d token skew at pos %d: go=(%d,%d) py=(%d,%d) text=%q",
					i, j, gotIDs[j], gotMask[j], tc.InputIDs[j], tc.AttentionMask[j], tc.Text)
			}
		}

		// Gate 3: FP32 & INT8 ONNX Logits Parity
		gotFP32, dFP32 := runSession(t, fp32Sess, gotIDs, gotMask)
		gotINT8, dINT8 := runSession(t, int8Sess, gotIDs, gotMask)
		fp32Total += dFP32
		int8Total += dINT8

		for c := range tc.FP32Logits {
			if diff := math.Abs(float64(gotFP32[c]) - tc.FP32Logits[c]); diff > 1e-4 {
				t.Fatalf("case %d fp32 logit[%d] skew: go=%.6f py=%.6f diff=%.6f", i, c, gotFP32[c], tc.FP32Logits[c], diff)
			}
			if diff := math.Abs(float64(gotINT8[c]) - tc.INT8Logits[c]); diff > 1e-3 {
				t.Fatalf("case %d int8 logit[%d] skew: go=%.6f py=%.6f diff=%.6f", i, c, gotINT8[c], tc.INT8Logits[c], diff)
			}
		}
		if gotLbl := contract.Labels[argmax(gotFP32)]; gotLbl != tc.ExpectedLabel {
			t.Fatalf("case %d fp32 label mismatch: go=%s py=%s", i, gotLbl, tc.ExpectedLabel)
		}
		if gotLbl := contract.Labels[argmax(gotINT8)]; gotLbl != tc.ExpectedLabel {
			t.Fatalf("case %d int8 label mismatch: go=%s py=%s", i, gotLbl, tc.ExpectedLabel)
		}
	}

	n := float64(len(golden.Cases))
	t.Logf("100/100 WordPiece + FP32 + INT8 Parity PASS | Go Avg Latency: FP32=%.3fms, INT8=%.3fms",
		float64(fp32Total.Microseconds())/1000.0/n,
		float64(int8Total.Microseconds())/1000.0/n,
	)
}
