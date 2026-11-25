package main_test

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/gunawanwijaya/apprentice/01-delivery-sla-breach/go/internal/contract"
	ort "github.com/yalue/onnxruntime_go"
)

type GoldenFile struct {
	SchemaVersion string  `json:"schema_version"`
	SchemaHash    string  `json:"schema_hash"`
	Threshold     float32 `json:"threshold"`
	Cases         []struct {
		Input         contract.PredictRequest `json:"input"`
		ExpectedProb  float64                 `json:"expected_prob"`
		ExpectedLabel int                     `json:"expected_label"`
	} `json:"cases"`
}

func TestPythonGoParity(t *testing.T) {
	// Point to your installed libonnxruntime.so / .dylib (e.g. from the Python .venv!)
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

	// Gate 1: Refuse to run if train.py and Go generated code drifted
	if golden.SchemaHash != contract.SchemaHash {
		t.Fatalf("schema drift! python=%s go=%s (re-run make gen)", golden.SchemaHash, contract.SchemaHash)
	}

	session, err := ort.NewDynamicAdvancedSession(
		"../artifacts/model.onnx",
		[]string{"input"},
		[]string{"label", "probabilities"},
		nil,
	)
	if err != nil {
		t.Fatalf("load model.onnx: %v", err)
	}
	defer session.Destroy()

	for i, tc := range golden.Cases {
		inTensor, err := ort.NewTensor(ort.NewShape(1, contract.FeatureCount), tc.Input.ToFloat32Slice())
		if err != nil {
			t.Fatalf("case %d input tensor: %v", i, err)
		}
		outputs := []ort.Value{nil, nil}
		err = session.Run([]ort.Value{inTensor}, outputs)
		inTensor.Destroy()
		if err != nil {
			t.Fatalf("case %d inference: %v", i, err)
		}

		probTensor := outputs[1].(*ort.Tensor[float32])
		gotProb := float64(probTensor.GetData()[1])
		outputs[0].Destroy()
		outputs[1].Destroy()

		if diff := math.Abs(gotProb - tc.ExpectedProb); diff > 1e-5 {
			t.Fatalf("case %d skew detected: go=%.6f py=%.6f diff=%.6f", i, gotProb, tc.ExpectedProb, diff)
		}
	}
}
