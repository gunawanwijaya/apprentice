package parity_test

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/gunawanwijaya/apprentice/07-agents-mcp-evals/go/internal/agent"
	"github.com/gunawanwijaya/apprentice/07-agents-mcp-evals/go/internal/contract"
	"github.com/gunawanwijaya/apprentice/07-agents-mcp-evals/go/internal/mcp"
)

type goldenFixtureFile struct {
	SchemaVersion string `json:"schema_version"`
	SchemaHash    string `json:"schema_hash"`
	OrdersCount   int    `json:"orders_count"`
	GoldenCases   []struct {
		CaseID                  string   `json:"case_id"`
		Category                string   `json:"category"`
		SessionTenant           string   `json:"session_tenant"`
		UserID                  string   `json:"user_id"`
		UserQuery               string   `json:"user_query"`
		RequestedRefundIDR      int64    `json:"requested_refund_idr"`
		SimulateDuplicateRetry  bool     `json:"simulate_duplicate_retry"`
		ExpectedDisposition     string   `json:"expected_disposition"`
		ExpectedTools           []string `json:"expected_tools"`
		ExpectedSteps           int      `json:"expected_steps"`
		MutatingBlocked         bool     `json:"mutating_blocked"`
		HITLEscalated           bool     `json:"hitl_escalated"`
		DuplicateShortCircuited bool     `json:"duplicate_short_circuited"`
	} `json:"golden_cases"`
}

func setupAgentAndFixtures(t *testing.T) (*mcp.Server, *agent.Agent, goldenFixtureFile) {
	t.Helper()
	root := ".."
	orders, err := mcp.LoadOrdersSeed(filepath.Join(root, "artifacts", "orders_seed.json"))
	if err != nil {
		t.Fatalf("load orders seed: %v", err)
	}
	policyMap := mcp.LoadPolicyFallbackMap(filepath.Join(root, "..", "06-production-rag", "artifacts", "corpus.jsonl"))
	srv := mcp.NewServer(orders, policyMap, "")

	raw, err := os.ReadFile(filepath.Join(root, "artifacts", "golden_fixtures.json"))
	if err != nil {
		t.Fatalf("read golden fixtures: %v", err)
	}
	var g goldenFixtureFile
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatalf("unmarshal golden fixtures: %v", err)
	}
	return srv, agent.New(srv), g
}

func TestPythonGoAgentParity(t *testing.T) {
	srv, ag, g := setupAgentAndFixtures(t)
	if contract.SchemaHash != g.SchemaHash {
		t.Fatalf("schema hash mismatch: go=%s python=%s", contract.SchemaHash, g.SchemaHash)
	}
	if srv.OrdersCount() != g.OrdersCount {
		t.Fatalf("orders count mismatch: go=%d python=%d", srv.OrdersCount(), g.OrdersCount)
	}

	ctx := context.Background()
	for _, tc := range g.GoldenCases {
		res, err := ag.Run(ctx, agent.RunRequest{
			SessionTenant:          tc.SessionTenant,
			UserID:                 tc.UserID,
			UserQuery:              tc.UserQuery,
			RequestedRefundIDR:     tc.RequestedRefundIDR,
			SimulateDuplicateRetry: tc.SimulateDuplicateRetry,
		})
		if err != nil {
			t.Fatalf("%s (%s) unexpected error: %v", tc.CaseID, tc.Category, err)
		}
		if res.Disposition != tc.ExpectedDisposition {
			t.Errorf("%s disposition mismatch: go=%s want=%s", tc.CaseID, res.Disposition, tc.ExpectedDisposition)
		}
		if !reflect.DeepEqual(res.Tools, tc.ExpectedTools) {
			t.Errorf("%s trajectory mismatch: go=%v want=%v", tc.CaseID, res.Tools, tc.ExpectedTools)
		}
		if res.Steps != tc.ExpectedSteps {
			t.Errorf("%s steps mismatch: go=%d want=%d", tc.CaseID, res.Steps, tc.ExpectedSteps)
		}
		if res.MutatingBlocked != tc.MutatingBlocked {
			t.Errorf("%s mutating_blocked mismatch: go=%t want=%t", tc.CaseID, res.MutatingBlocked, tc.MutatingBlocked)
		}
		if res.HITLEscalated != tc.HITLEscalated {
			t.Errorf("%s hitl_escalated mismatch: go=%t want=%t", tc.CaseID, res.HITLEscalated, tc.HITLEscalated)
		}
		if res.DuplicateShortCircuited != tc.DuplicateShortCircuited {
			t.Errorf("%s duplicate_short_circuited mismatch: go=%t want=%t", tc.CaseID, res.DuplicateShortCircuited, tc.DuplicateShortCircuited)
		}
		if res.Steps > contract.MaxSteps {
			t.Errorf("%s exceeded max_steps=%d: got %d", tc.CaseID, contract.MaxSteps, res.Steps)
		}
	}
}

func TestMCPJSONRPCAndSchemaSafety(t *testing.T) {
	srv, _, _ := setupAgentAndFixtures(t)

	// 1. Verify tools/list never exposes tenant_id or user_id to the LLM
	listBody := `{"jsonrpc":"2.0","id":1,"method":"tools/list"}`
	req := httptest.NewRequest(http.MethodPost, "/mcp", strings.NewReader(listBody))
	req.Header.Set("X-Tenant-Id", "ORDER")
	rec := httptest.NewRecorder()
	srv.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("tools/list status=%d", rec.Code)
	}
	if bytes.Contains(rec.Body.Bytes(), []byte(`"tenant_id"`)) || bytes.Contains(rec.Body.Bytes(), []byte(`"user_id"`)) {
		t.Fatalf("confused-deputy vulnerability: tools/list exposed tenant_id or user_id in LLM inputSchema")
	}

	// 2. Verify cross-tenant tools/call is blocked by session header
	callBody := `{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_order_dossier","arguments":{"order_id":"ORD-9001"}}}`
	req2 := httptest.NewRequest(http.MethodPost, "/mcp", strings.NewReader(callBody))
	req2.Header.Set("X-Tenant-Id", "ORDER") // ORD-9001 belongs to ACCOUNT
	rec2 := httptest.NewRecorder()
	srv.ServeHTTP(rec2, req2)
	if !bytes.Contains(rec2.Body.Bytes(), []byte("forbidden_cross_tenant")) {
		t.Fatalf("expected forbidden_cross_tenant in MCP response, got: %s", rec2.Body.String())
	}
}

func TestJudgeRubricValidation(t *testing.T) {
	if err := agent.ValidateJudgeVerdict(agent.JudgeVerdict{
		Reasoning:            "",
		HallucinatedFact:     false,
		PolicyCompliant:      true,
		AnsweredUserQuestion: true,
		Pass:                 true,
	}); err == nil {
		t.Fatalf("expected error when reasoning is missing before pass verdict")
	}

	if err := agent.ValidateJudgeVerdict(agent.JudgeVerdict{
		Reasoning:            "Cited policy chunk POL-ORDE-01#0 accurately and respected the 250k IDR auto-refund cap.",
		HallucinatedFact:     false,
		PolicyCompliant:      true,
		AnsweredUserQuestion: true,
		Pass:                 true,
	}); err != nil {
		t.Fatalf("expected valid CoT binary verdict, got: %v", err)
	}
}
