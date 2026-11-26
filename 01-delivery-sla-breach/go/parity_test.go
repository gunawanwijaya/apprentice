package main_test

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/gunawanwijaya/apprentice/01-delivery-sla-breach/go/internal/contract"
	ort "github.com/yalue/onnxruntime_go"
)

type CQRCalibration struct {
	CoverageTarget float64 `json:"coverage_target"`
	CQRMarginDays  float64 `json:"cqr_margin_days"`
}

type GoldenFile struct {
	SchemaVersion string  `json:"schema_version"`
	SchemaHash    string  `json:"schema_hash"`
	Threshold     float32 `json:"threshold"`
	Cases         []struct {
		Input              contract.PredictRequest `json:"input"`
		ExpectedProb       float64                 `json:"expected_prob"`
		ExpectedLabel      int                     `json:"expected_label"`
		ExpectedETAMinDays float64                 `json:"expected_eta_min_days"`
		ExpectedETAP50Days float64                 `json:"expected_eta_p50_days"`
		ExpectedETAMaxDays float64                 `json:"expected_eta_max_days"`
	} `json:"cases"`
}

func runScalarONNX(t *testing.T, sess *ort.DynamicAdvancedSession, inTensor *ort.Tensor[float32]) float64 {
	t.Helper()
	out := []ort.Value{nil}
	if err := sess.Run([]ort.Value{inTensor}, out); err != nil {
		t.Fatalf("scalar onnx run: %v", err)
	}
	defer out[0].Destroy()
	return float64(out[0].(*ort.Tensor[float32]).GetData()[0])
}

func TestPythonGoParity(t *testing.T) {
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

	calRaw, err := os.ReadFile("../artifacts/cqr_calibration.json")
	if err != nil {
		t.Fatalf("read cqr_calibration.json: %v", err)
	}
	var cal CQRCalibration
	if err := json.Unmarshal(calRaw, &cal); err != nil {
		t.Fatalf("unmarshal cqr_calibration.json: %v", err)
	}

	clfSess, err := ort.NewDynamicAdvancedSession("../artifacts/model.onnx", []string{"input"}, []string{"label", "probabilities"}, nil)
	if err != nil {
		t.Fatalf("load model.onnx: %v", err)
	}
	defer clfSess.Destroy()

	p10Sess, err := ort.NewDynamicAdvancedSession("../artifacts/eta_p10.onnx", []string{"input"}, []string{"variable"}, nil)
	if err != nil {
		t.Fatalf("load eta_p10.onnx: %v", err)
	}
	defer p10Sess.Destroy()

	p50Sess, err := ort.NewDynamicAdvancedSession("../artifacts/eta_p50.onnx", []string{"input"}, []string{"variable"}, nil)
	if err != nil {
		t.Fatalf("load eta_p50.onnx: %v", err)
	}
	defer p50Sess.Destroy()

	p90Sess, err := ort.NewDynamicAdvancedSession("../artifacts/eta_p90.onnx", []string{"input"}, []string{"variable"}, nil)
	if err != nil {
		t.Fatalf("load eta_p90.onnx: %v", err)
	}
	defer p90Sess.Destroy()

	for i, tc := range golden.Cases {
		inTensor, err := ort.NewTensor(ort.NewShape(1, contract.FeatureCount), tc.Input.ToFloat32Slice())
		if err != nil {
			t.Fatalf("case %d input tensor: %v", i, err)
		}

		clfOut := []ort.Value{nil, nil}
		if err := clfSess.Run([]ort.Value{inTensor}, clfOut); err != nil {
			inTensor.Destroy()
			t.Fatalf("case %d clf inference: %v", i, err)
		}
		gotProb := float64(clfOut[1].(*ort.Tensor[float32]).GetData()[1])
		clfOut[0].Destroy()
		clfOut[1].Destroy()

		rawP10 := runScalarONNX(t, p10Sess, inTensor)
		rawP50 := runScalarONNX(t, p50Sess, inTensor)
		rawP90 := runScalarONNX(t, p90Sess, inTensor)
		inTensor.Destroy()

		etaMin := math.Max(1.0, rawP10-cal.CQRMarginDays)
		etaMax := math.Max(etaMin, rawP90+cal.CQRMarginDays)
		etaP50 := math.Min(math.Max(rawP50, etaMin), etaMax)

		if diff := math.Abs(gotProb - tc.ExpectedProb); diff > 1e-5 {
			t.Fatalf("case %d prob skew: go=%.6f py=%.6f", i, gotProb, tc.ExpectedProb)
		}
		if math.Abs(etaMin-tc.ExpectedETAMinDays) > 1e-4 ||
			math.Abs(etaP50-tc.ExpectedETAP50Days) > 1e-4 ||
			math.Abs(etaMax-tc.ExpectedETAMaxDays) > 1e-4 {
			t.Fatalf("case %d cqr skew: go=[%.4f, %.4f, %.4f] py=[%.4f, %.4f, %.4f]",
				i, etaMin, etaP50, etaMax, tc.ExpectedETAMinDays, tc.ExpectedETAP50Days, tc.ExpectedETAMaxDays)
		}
	}
}
