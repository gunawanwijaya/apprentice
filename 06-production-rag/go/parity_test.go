package parity_test

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/gunawanwijaya/apprentice/06-production-rag/go/internal/contract"
	"github.com/gunawanwijaya/apprentice/06-production-rag/go/internal/rag"
)

type goldenFile struct {
	SchemaVersion string `json:"schema_version"`
	SchemaHash    string `json:"schema_hash"`
	CorpusSize    int    `json:"corpus_size"`
	GoldenCases   []struct {
		Query           string  `json:"query"`
		TenantID        string  `json:"tenant_id"`
		QueryType       string  `json:"query_type"`
		ExpectedChunkID string  `json:"expected_chunk_id"`
		TopRRFChunkID   string  `json:"top_rrf_chunk_id"`
		TopRRFScore     float64 `json:"top_rrf_score"`
	} `json:"golden_cases"`
}

func loadTestEngine(t *testing.T) (*rag.Engine, goldenFile) {
	t.Helper()
	root := ".."
	c, hash, err := rag.LoadContract(filepath.Join(root, "feature_contract.json"))
	if err != nil {
		t.Fatalf("load contract: %v", err)
	}
	docs, err := rag.LoadCorpusJSONL(filepath.Join(root, "artifacts", "corpus.jsonl"))
	if err != nil {
		t.Fatalf("load corpus: %v", err)
	}
	raw, err := os.ReadFile(filepath.Join(root, "artifacts", "golden_fixtures.json"))
	if err != nil {
		t.Fatalf("read golden fixtures: %v", err)
	}
	var g goldenFile
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("unmarshal golden fixtures: %v", err)
	}
	return rag.NewEngine(c, hash, docs), g
}

func TestPythonGoParity(t *testing.T) {
	eng, g := loadTestEngine(t)
	if contract.SchemaHash != eng.SchemaHash {
		t.Fatalf("codegen hash mismatch: codegen=%s runtime=%s", contract.SchemaHash, eng.SchemaHash)
	}
	if eng.SchemaHash != g.SchemaHash {
		t.Fatalf("schema hash mismatch: go=%s python=%s", eng.SchemaHash, g.SchemaHash)
	}
	if eng.CorpusSize() != g.CorpusSize {
		t.Fatalf("corpus size mismatch: go=%d python=%d", eng.CorpusSize(), g.CorpusSize)
	}

	ctx := context.Background()
	for i, tc := range g.GoldenCases {
		hits, err := eng.Retrieve(ctx, tc.Query, tc.TenantID)
		if err != nil {
			t.Fatalf("case %d retrieve error: %v", i, err)
		}
		if len(hits) == 0 {
			t.Fatalf("case %d returned 0 hits for query %q", i, tc.Query)
		}
		if hits[0].Chunk.ChunkID != tc.TopRRFChunkID {
			t.Errorf("case %d (%s) top chunk mismatch: go=%s python=%s",
				i, tc.QueryType, hits[0].Chunk.ChunkID, tc.TopRRFChunkID)
		}
		if math.Abs(hits[0].Score-tc.TopRRFScore) > 1e-5 {
			t.Errorf("case %d (%s) top score mismatch: go=%.6f python=%.6f",
				i, tc.QueryType, hits[0].Score, tc.TopRRFScore)
		}
	}
}

func TestTenantIsolationAndCitations(t *testing.T) {
	eng, _ := loadTestEngine(t)
	ctx := context.Background()

	if _, err := eng.Retrieve(ctx, "cancel my order", ""); err == nil {
		t.Fatalf("expected error when tenant_id is empty")
	}

	hits, err := eng.Retrieve(ctx, "how do I cancel my order", "ORDER")
	if err != nil {
		t.Fatalf("retrieve ORDER: %v", err)
	}
	for _, h := range hits {
		if h.Chunk.TenantID != "ORDER" {
			t.Fatalf("tenant leak detected: got %s, want ORDER", h.Chunk.TenantID)
		}
	}

	goodAnswer := "Follow the policy in [" + hits[0].Chunk.ChunkID + "]."
	if ok, _, inv := rag.VerifyCitations(goodAnswer, hits); !ok || len(inv) > 0 {
		t.Fatalf("expected valid citation, got ok=%v invalid=%v", ok, inv)
	}
	badAnswer := "According to [POL-FAKE-99#0], you get a free car."
	if ok, _, inv := rag.VerifyCitations(badAnswer, hits); ok || len(inv) != 1 {
		t.Fatalf("expected invalid citation detection, got ok=%v invalid=%v", ok, inv)
	}
}
