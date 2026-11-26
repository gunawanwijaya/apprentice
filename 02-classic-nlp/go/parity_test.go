package main_test

import (
	"encoding/json"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/gunawanwijaya/apprentice/02-classic-nlp/go/internal/bm25"
	"github.com/gunawanwijaya/apprentice/02-classic-nlp/go/internal/contract"
	"github.com/gunawanwijaya/apprentice/02-classic-nlp/go/internal/pii"
	"github.com/gunawanwijaya/apprentice/02-classic-nlp/go/internal/viterbi"
	ort "github.com/yalue/onnxruntime_go"
)

type GoldenFile struct {
	SchemaVersion string `json:"schema_version"`
	SchemaHash    string `json:"schema_hash"`
	Intent        []struct {
		Text          string  `json:"text"`
		ExpectedLabel string  `json:"expected_label"`
		ExpectedProb  float64 `json:"expected_prob"`
	} `json:"intent"`
	BM25 []struct {
		Query          string   `json:"query"`
		ExpectedTopIDs []string `json:"expected_top_ids"`
	} `json:"bm25"`
	NER []struct {
		Tokens       []string `json:"tokens"`
		ExpectedTags []string `json:"expected_tags"`
	} `json:"ner"`
	Triage []struct {
		Text                 string  `json:"text"`
		ExpectedQueue        string  `json:"expected_queue"`
		ExpectedQueueProb    float64 `json:"expected_queue_prob"`
		ExpectedPriority     string  `json:"expected_priority"`
		ExpectedPriorityProb float64 `json:"expected_priority_prob"`
	} `json:"triage"`
}

func runTextONNX(t *testing.T, sess *ort.DynamicAdvancedSession, text string) (string, float64) {
	t.Helper()
	inTensor, err := ort.NewStringTensor(ort.NewShape(1, 1))
	if err != nil {
		t.Fatalf("string tensor: %v", err)
	}
	defer inTensor.Destroy()

	if err := inTensor.SetContents([]string{text}); err != nil {
		t.Fatalf("set string contents: %v", err)
	}

	outputs := []ort.Value{nil, nil}
	if err := sess.Run([]ort.Value{inTensor}, outputs); err != nil {
		t.Fatalf("onnx run: %v", err)
	}
	defer outputs[0].Destroy()
	defer outputs[1].Destroy()

	lbls, err := outputs[0].(*ort.StringTensor).GetContents()
	if err != nil {
		t.Fatalf("read label tensor: %v", err)
	}
	probs := outputs[1].(*ort.Tensor[float32]).GetData()
	maxP := float32(0)
	for _, p := range probs {
		if p > maxP {
			maxP = p
		}
	}
	return lbls[0], float64(maxP)
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

	t.Run("intent_onnx", func(t *testing.T) {
		sess, err := ort.NewDynamicAdvancedSession("../artifacts/intent.onnx", []string{"input"}, []string{"label", "probabilities"}, nil)
		if err != nil {
			t.Fatalf("load intent.onnx: %v", err)
		}
		defer sess.Destroy()

		for i, tc := range golden.Intent {
			gotLbl, gotProb := runTextONNX(t, sess, tc.Text)
			if gotLbl != tc.ExpectedLabel || math.Abs(gotProb-tc.ExpectedProb) > 1e-4 {
				t.Fatalf("case %d skew: got=(%s, %.6f) want=(%s, %.6f)", i, gotLbl, gotProb, tc.ExpectedLabel, tc.ExpectedProb)
			}
		}
	})

	t.Run("bm25_pure_go", func(t *testing.T) {
		idx, err := bm25.Load("../artifacts/bm25_corpus.json", contract.TokenPattern)
		if err != nil {
			t.Fatalf("load bm25_corpus.json: %v", err)
		}
		for i, tc := range golden.BM25 {
			res := idx.Search(tc.Query, contract.BM25TopK)
			gotIDs := make([]string, len(res))
			for j, r := range res {
				gotIDs[j] = r.ID
			}
			if !reflect.DeepEqual(gotIDs, tc.ExpectedTopIDs) {
				t.Fatalf("case %d bm25 mismatch: got=%v want=%v", i, gotIDs, tc.ExpectedTopIDs)
			}
		}
	})

	t.Run("ner_crf_viterbi", func(t *testing.T) {
		m, err := viterbi.Load("../artifacts/ner_crf.json")
		if err != nil {
			t.Fatalf("load ner_crf.json: %v", err)
		}
		for i, tc := range golden.NER {
			got := m.Predict(tc.Tokens)
			if !reflect.DeepEqual(got, tc.ExpectedTags) {
				t.Fatalf("case %d ner mismatch:\ntokens=%v\ngot =%v\nwant=%v", i, tc.Tokens, got, tc.ExpectedTags)
			}
		}
	})

	t.Run("triage_onnx", func(t *testing.T) {
		qSess, err := ort.NewDynamicAdvancedSession("../artifacts/queue.onnx", []string{"input"}, []string{"label", "probabilities"}, nil)
		if err != nil {
			t.Fatalf("load queue.onnx: %v", err)
		}
		defer qSess.Destroy()

		pSess, err := ort.NewDynamicAdvancedSession("../artifacts/priority.onnx", []string{"input"}, []string{"label", "probabilities"}, nil)
		if err != nil {
			t.Fatalf("load priority.onnx: %v", err)
		}
		defer pSess.Destroy()

		for i, tc := range golden.Triage {
			gotQ, gotQP := runTextONNX(t, qSess, tc.Text)
			gotP, gotPP := runTextONNX(t, pSess, tc.Text)
			if gotQ != tc.ExpectedQueue || math.Abs(gotQP-tc.ExpectedQueueProb) > 1e-4 {
				t.Fatalf("case %d queue skew: got=(%s, %.6f) want=(%s, %.6f)", i, gotQ, gotQP, tc.ExpectedQueue, tc.ExpectedQueueProb)
			}
			if gotP != tc.ExpectedPriority || math.Abs(gotPP-tc.ExpectedPriorityProb) > 1e-4 {
				t.Fatalf("case %d priority skew: got=(%s, %.6f) want=(%s, %.6f)", i, gotP, gotPP, tc.ExpectedPriority, tc.ExpectedPriorityProb)
			}
		}
	})

	t.Run("pii_redactor", func(t *testing.T) {
		in := "Hubungi 081234567890 atau +6281703020777, NIK 3171012345678901, email budi@example.com"
		want := "Hubungi [PHONE] atau [PHONE], NIK [KTP], email [EMAIL]"
		if got := pii.Redact(in); got != want {
			t.Fatalf("pii mismatch:\ngot =%s\nwant=%s", got, want)
		}
	})
}
