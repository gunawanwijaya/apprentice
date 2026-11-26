package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/gunawanwijaya/apprentice/05-llm-gateway/go/internal/cache"
	"github.com/gunawanwijaya/apprentice/05-llm-gateway/go/internal/contract"
	"github.com/gunawanwijaya/apprentice/05-llm-gateway/go/internal/guard"
	"github.com/gunawanwijaya/apprentice/05-llm-gateway/go/internal/ledger"
	"github.com/gunawanwijaya/apprentice/05-llm-gateway/go/internal/provider"
)

func setupTestGateway(t *testing.T) (*httptest.Server, *ledger.Manager, *provider.MockUpstreamStats) {
	t.Helper()
	var stats provider.MockUpstreamStats
	pri := httptest.NewServer(provider.NewMockUpstreamHandler("openai", 5*time.Millisecond, 12*time.Millisecond, &stats))
	sec := httptest.NewServer(provider.NewMockUpstreamHandler("gemini", 4*time.Millisecond, 6*time.Millisecond, &stats))
	t.Cleanup(func() {
		pri.Close()
		sec.Close()
	})

	hc := &http.Client{Timeout: 10 * time.Second}
	router := &provider.Router{
		Primary:   &provider.OpenAIHTTPClient{ProviderName: "openai", BaseURL: pri.URL, HTTPClient: hc},
		Secondary: &provider.OpenAIHTTPClient{ProviderName: "gemini", BaseURL: sec.URL, HTTPClient: hc},
	}
	store := cache.NewStore(contract.SemanticSimilarityThreshold)
	mgr := ledger.NewManager(contract.DefaultRPM, contract.DefaultDailyBudgetUSD)
	gw := httptest.NewServer(NewGatewayMux(router, store, mgr))
	t.Cleanup(gw.Close)
	return gw, mgr, &stats
}

func postJSON(t *testing.T, url string, body any) (*http.Response, []byte) {
	t.Helper()
	b, _ := json.Marshal(body)
	resp, err := http.Post(url, "application/json", bytes.NewReader(b))
	if err != nil {
		t.Fatalf("POST %s: %v", url, err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	return resp, raw
}

func TestGoldenParityAndEntityGuard(t *testing.T) {
	raw, err := os.ReadFile("../../../artifacts/golden_fixtures.json")
	if err != nil {
		t.Fatalf("read golden_fixtures.json: %v", err)
	}
	var golden struct {
		SchemaHash string `json:"schema_hash"`
		GuardCases []struct {
			Prompt          string `json:"prompt"`
			EntitySensitive bool   `json:"entity_sensitive"`
		} `json:"guard_cases"`
	}
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatalf("unmarshal golden_fixtures.json: %v", err)
	}
	if golden.SchemaHash != contract.SchemaHash {
		t.Fatalf("schema hash drift: python=%s go=%s", golden.SchemaHash, contract.SchemaHash)
	}
	for i, tc := range golden.GuardCases {
		got := guard.Inspect(tc.Prompt)
		if got.EntitySensitive != tc.EntitySensitive {
			t.Fatalf("guard case %d (%q): got=%v want=%v (tags=%v)", i, tc.Prompt, got.EntitySensitive, tc.EntitySensitive, got.MatchedTags)
		}
	}
}

func TestSemanticCacheHitVsOrderIDTrapPrevention(t *testing.T) {
	gw, _, _ := setupTestGateway(t)

	_, raw1 := postJSON(t, gw.URL+"/v1/chat/completions", GatewayChatRequest{
		TenantID:    "toko-helpdesk",
		Temperature: 0.2,
		Messages:    []provider.Message{{Role: "user", Content: "How do I reset my account security PIN?"}},
	})
	var r1 GatewayChatResponse
	_ = json.Unmarshal(raw1, &r1)
	if r1.CacheStatus != "miss" || r1.EntitySensitive {
		t.Fatalf("expected clean FAQ miss, got status=%s entity=%v", r1.CacheStatus, r1.EntitySensitive)
	}

	_, raw2 := postJSON(t, gw.URL+"/v1/chat/completions", GatewayChatRequest{
		TenantID:    "toko-helpdesk",
		Temperature: 0.2,
		Messages:    []provider.Message{{Role: "user", Content: "Please tell me how to reset my security PIN"}},
	})
	var r2 GatewayChatResponse
	_ = json.Unmarshal(raw2, &r2)
	if r2.CacheStatus != "semantic_vector" || r2.CostUSD != 0 {
		t.Fatalf("expected semantic_vector cache hit ($0), got status=%s sim=%.3f", r2.CacheStatus, r2.Similarity)
	}

	_, ord1 := postJSON(t, gw.URL+"/v1/chat/completions", GatewayChatRequest{
		TenantID:    "fintech-ops",
		Temperature: 0.2,
		Messages:    []provider.Message{{Role: "user", Content: "Check refund eligibility for Order #10042"}},
	})
	_, ord2 := postJSON(t, gw.URL+"/v1/chat/completions", GatewayChatRequest{
		TenantID:    "fintech-ops",
		Temperature: 0.2,
		Messages:    []provider.Message{{Role: "user", Content: "Check refund eligibility for Order #10049"}},
	})
	var o1, o2 GatewayChatResponse
	_ = json.Unmarshal(ord1, &o1)
	_ = json.Unmarshal(ord2, &o2)
	if !o1.EntitySensitive || !o2.EntitySensitive || o2.CacheStatus != "miss" {
		t.Fatalf("semantic cache trap failed! o1_entity=%v o2_entity=%v o2_cache=%s", o1.EntitySensitive, o2.EntitySensitive, o2.CacheStatus)
	}
}

func TestFailoverAndStructuredJSONRepair(t *testing.T) {
	gw, _, _ := setupTestGateway(t)

	resp, raw := postJSON(t, gw.URL+"/v1/chat/completions", GatewayChatRequest{
		TenantID:    "tenant-failover",
		FaultInject: "primary_429",
		Messages:    []provider.Message{{Role: "user", Content: "Summarize SLA status"}},
	})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 after failover, got %d: %s", resp.StatusCode, string(raw))
	}
	var chatResp GatewayChatResponse
	_ = json.Unmarshal(raw, &chatResp)
	if !chatResp.FailedOver || chatResp.Model != contract.FallbackModel {
		t.Fatalf("expected failover to %s, got failed_over=%v model=%s", contract.FallbackModel, chatResp.FailedOver, chatResp.Model)
	}

	extResp, extRaw := postJSON(t, gw.URL+"/v1/extract/refund", RefundExtractRequest{
		TenantID:         "tenant-extract",
		TicketText:       "Customer parcel crushed in transit for Order #10042",
		MaxRefundableIDR: 200000,
		FaultInject:      "bad_json_once",
	})
	if extResp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 after repair retry, got %d: %s", extResp.StatusCode, string(extRaw))
	}
	var parsed struct {
		Decision RefundDecision `json:"decision"`
		Attempts int            `json:"attempts"`
	}
	_ = json.Unmarshal(extRaw, &parsed)
	if parsed.Attempts != 2 || parsed.Decision.Verdict != "APPROVE" || parsed.Decision.OrderID != "10042" {
		t.Fatalf("unexpected repaired extraction: %+v", parsed)
	}
}

func TestSSEContextCancellationAndMidstreamError(t *testing.T) {
	gw, mgr, stats := setupTestGateway(t)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	reqBody, _ := json.Marshal(GatewayChatRequest{
		TenantID: "mobile-sse",
		Stream:   true,
		Messages: []provider.Message{{Role: "user", Content: "Stream a multi-token explanation"}},
	})
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, gw.URL+"/v1/chat/completions", bytes.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("SSE request failed: %v", err)
	}
	scanner := bufio.NewScanner(resp.Body)
	seen := 0
	for scanner.Scan() {
		if scanner.Err() != nil {
			cancel()
			break
		}
		if strings.HasPrefix(scanner.Text(), "data:") {
			seen++
			if seen == 2 {
				cancel()
				break
			}
		}
	}
	_ = resp.Body.Close()
	time.Sleep(40 * time.Millisecond)

	if stats.AbortedStreams.Load() < 1 {
		t.Fatalf("expected upstream mock to observe context cancellation abort, got %d", stats.AbortedStreams.Load())
	}
	snap := mgr.Snapshot("mobile-sse")
	if snap.StreamAborts != 1 || snap.CompletionTokens == 0 || snap.SpendUSD <= 0 {
		t.Fatalf("expected defer ledger to record aborted stream tokens, got %+v", snap)
	}

	reqBody2, _ := json.Marshal(GatewayChatRequest{
		TenantID:    "mobile-cut",
		Stream:      true,
		FaultInject: "midstream_cut",
		Messages:    []provider.Message{{Role: "user", Content: "Trigger midstream upstream disconnect"}},
	})
	resp2, err := http.Post(gw.URL+"/v1/chat/completions", "application/json", bytes.NewReader(reqBody2))
	if err != nil {
		t.Fatalf("SSE cut request: %v", err)
	}
	defer resp2.Body.Close()
	allSSE, _ := io.ReadAll(resp2.Body)
	if !strings.Contains(string(allSSE), "event: error") || !strings.Contains(string(allSSE), "upstream_stream_interrupted") {
		t.Fatalf("expected typed SSE error event after HTTP 200, got:\n%s", string(allSSE))
	}

	mgr.SetTenantPolicy("broke-tenant", 60, 0.0000001)
	_, _ = postJSON(t, gw.URL+"/v1/chat/completions", GatewayChatRequest{
		TenantID: "broke-tenant",
		Messages: []provider.Message{{Role: "user", Content: "First call exhausts tiny budget"}},
	})
	bResp, _ := postJSON(t, gw.URL+"/v1/chat/completions", GatewayChatRequest{
		TenantID: "broke-tenant",
		Messages: []provider.Message{{Role: "user", Content: "Second call must be blocked"}},
	})
	if bResp.StatusCode != http.StatusPaymentRequired {
		t.Fatalf("expected 402 Payment Required on budget exhaustion, got %d", bResp.StatusCode)
	}
}
