package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/gunawanwijaya/apprentice/05-llm-gateway/go/internal/cache"
	"github.com/gunawanwijaya/apprentice/05-llm-gateway/go/internal/contract"
	"github.com/gunawanwijaya/apprentice/05-llm-gateway/go/internal/guard"
	"github.com/gunawanwijaya/apprentice/05-llm-gateway/go/internal/ledger"
	"github.com/gunawanwijaya/apprentice/05-llm-gateway/go/internal/provider"
)

type GatewayChatRequest struct {
	TenantID    string             `json:"tenant_id"`
	Model       string             `json:"model,omitempty"`
	Messages    []provider.Message `json:"messages"`
	Temperature float64            `json:"temperature"`
	MaxTokens   int                `json:"max_tokens,omitempty"`
	Stream      bool               `json:"stream,omitempty"`
	FaultInject string             `json:"fault_inject,omitempty"`
}

type GatewayChatResponse struct {
	Model            string   `json:"model"`
	Provider         string   `json:"provider"`
	Content          string   `json:"content"`
	CacheStatus      string   `json:"cache_status"`
	Similarity       float64  `json:"similarity,omitempty"`
	EntitySensitive  bool     `json:"entity_sensitive"`
	MatchedEntities  []string `json:"matched_entities,omitempty"`
	FailedOver       bool     `json:"failed_over"`
	PromptTokens     int      `json:"prompt_tokens"`
	CompletionTokens int      `json:"completion_tokens"`
	CostUSD          float64  `json:"cost_usd"`
	LatencyUs        int64    `json:"latency_us"`
	SchemaHash       string   `json:"schema_hash"`
}

type RefundExtractRequest struct {
	TenantID         string  `json:"tenant_id"`
	TicketText       string  `json:"ticket_text"`
	MaxRefundableIDR float64 `json:"max_refundable_idr"`
	FaultInject      string  `json:"fault_inject,omitempty"`
}

type RefundDecision struct {
	OrderID     string  `json:"order_id"`
	Verdict     string  `json:"verdict"`
	ApprovedIDR float64 `json:"approved_idr"`
	ReasonCode  string  `json:"reason_code"`
}

func NewGatewayMux(router *provider.Router, cacheStore *cache.Store, ledgerMgr *ledger.Manager) *http.ServeMux {
	mux := http.NewServeMux()

	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"status":         "ok",
			"schema_version": contract.SchemaVersion,
			"schema_hash":    contract.SchemaHash,
			"primary_model":  contract.PrimaryModel,
			"fallback_model": contract.FallbackModel,
		})
	})

	mux.HandleFunc("GET /v1/ledger", func(w http.ResponseWriter, r *http.Request) {
		tenant := r.URL.Query().Get("tenant_id")
		if tenant == "" {
			tenant = "default"
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(ledgerMgr.Snapshot(tenant))
	})

	mux.HandleFunc("POST /v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		var req GatewayChatRequest
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&req); err != nil {
			http.Error(w, `{"error":"invalid request json"}`, http.StatusBadRequest)
			return
		}
		if req.TenantID == "" {
			req.TenantID = "default"
		}
		if req.Model == "" {
			req.Model = contract.PrimaryModel
		}
		if req.MaxTokens <= 0 || req.MaxTokens > contract.MaxOutputTokensCap {
			req.MaxTokens = contract.MaxOutputTokensCap
		}

		if err := ledgerMgr.CheckAdmission(req.TenantID); err != nil {
			code := http.StatusTooManyRequests
			if errors.Is(err, ledger.ErrBudgetExceeded) {
				code = http.StatusPaymentRequired
			}
			http.Error(w, fmt.Sprintf(`{"error":%q}`, err.Error()), code)
			return
		}

		var userBuf strings.Builder
		var fullBuf strings.Builder
		for _, m := range req.Messages {
			fullBuf.WriteString(m.Role)
			fullBuf.WriteByte(':')
			fullBuf.WriteString(m.Content)
			fullBuf.WriteByte('\n')
			if m.Role == "user" {
				userBuf.WriteString(m.Content)
				userBuf.WriteByte(' ')
			}
		}
		userText := strings.TrimSpace(userBuf.String())
		scan := guard.Inspect(userText)

		allowExact := !req.Stream && req.Temperature <= 0.1
		allowSemantic := !req.Stream && (!contract.BlockSemanticCacheOnEntity || !scan.EntitySensitive)
		exactKey := cache.ExactKey(req.TenantID, req.Model, req.Temperature, fullBuf.String(), "")

		if !req.Stream {
			if hit, status, sim, ok := cacheStore.Lookup(req.TenantID, exactKey, userText, allowExact, allowSemantic); ok {
				ledgerMgr.Record(req.TenantID, 0, 0, 0.0, true, false, false)
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(GatewayChatResponse{
					Model:           hit.Model,
					Provider:        "gateway_cache",
					Content:         hit.Content,
					CacheStatus:     status,
					Similarity:      sim,
					EntitySensitive: scan.EntitySensitive,
					MatchedEntities: scan.MatchedTags,
					CostUSD:         0.0,
					LatencyUs:       time.Since(start).Microseconds(),
					SchemaHash:      contract.SchemaHash,
				})
				return
			}

			res, err := router.CompleteWithFailover(r.Context(), provider.ChatRequest{
				Model:       req.Model,
				Messages:    req.Messages,
				Temperature: req.Temperature,
				MaxTokens:   req.MaxTokens,
				FaultInject: req.FaultInject,
			})
			if err != nil {
				var upErr *provider.UpstreamError
				if errors.As(err, &upErr) && upErr.StatusCode == http.StatusBadRequest {
					http.Error(w, fmt.Sprintf(`{"error":%q}`, upErr.Message), http.StatusBadRequest)
					return
				}
				http.Error(w, fmt.Sprintf(`{"error":%q}`, err.Error()), http.StatusBadGateway)
				return
			}

			cost := contract.ComputeCostUSD(res.Model, res.PromptTokens, res.CompletionTokens)
			ledgerMgr.Record(req.TenantID, res.PromptTokens, res.CompletionTokens, cost, false, false, res.FailedOver)
			cacheStore.Put(req.TenantID, exactKey, userText, cache.CachedResponse{
				Content:          res.Content,
				Model:            res.Model,
				PromptTokens:     res.PromptTokens,
				CompletionTokens: res.CompletionTokens,
			}, allowExact, allowSemantic)

			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(GatewayChatResponse{
				Model:            res.Model,
				Provider:         res.Provider,
				Content:          res.Content,
				CacheStatus:      "miss",
				EntitySensitive:  scan.EntitySensitive,
				MatchedEntities:  scan.MatchedTags,
				FailedOver:       res.FailedOver,
				PromptTokens:     res.PromptTokens,
				CompletionTokens: res.CompletionTokens,
				CostUSD:          cost,
				LatencyUs:        time.Since(start).Microseconds(),
				SchemaHash:       contract.SchemaHash,
			})
			return
		}

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, `{"error":"streaming unsupported"}`, http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.WriteHeader(http.StatusOK)

		promptToks := max(8, fullBuf.Len()/4)
		streamedChunks := 0
		exactUsageReceived := false
		aborted := false

		defer func() {
			cost := contract.ComputeCostUSD(req.Model, promptToks, streamedChunks)
			ledgerMgr.Record(req.TenantID, promptToks, streamedChunks, cost, false, aborted, false)
		}()

		err := router.Primary.Stream(r.Context(), req.Model, provider.ChatRequest{
			Model:       req.Model,
			Messages:    req.Messages,
			Temperature: req.Temperature,
			MaxTokens:   req.MaxTokens,
			Stream:      true,
			FaultInject: req.FaultInject,
		}, func(d provider.StreamDelta) error {
			if d.PromptTokens > 0 || d.CompletionTokens > 0 {
				promptToks = d.PromptTokens
				streamedChunks = d.CompletionTokens
				exactUsageReceived = true
			}
			if d.Token != "" {
				if !exactUsageReceived {
					streamedChunks++
				}
				payload, _ := json.Marshal(map[string]string{"delta": d.Token})
				_, _ = fmt.Fprintf(w, "data: %s\n\n", payload)
				flusher.Flush()
			}
			if d.Done {
				_, _ = fmt.Fprintf(w, "data: [DONE]\n\n")
				flusher.Flush()
			}
			return nil
		})

		if err != nil {
			aborted = true
			if r.Context().Err() == nil {
				errPayload, _ := json.Marshal(map[string]any{
					"code":      "upstream_stream_interrupted",
					"message":   err.Error(),
					"retryable": true,
				})
				_, _ = fmt.Fprintf(w, "event: error\ndata: %s\n\n", errPayload)
				flusher.Flush()
			}
		}
	})

	mux.HandleFunc("POST /v1/extract/refund", func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		var req RefundExtractRequest
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<18)).Decode(&req); err != nil {
			http.Error(w, `{"error":"invalid request json"}`, http.StatusBadRequest)
			return
		}
		if req.TenantID == "" {
			req.TenantID = "default"
		}
		if err := ledgerMgr.CheckAdmission(req.TenantID); err != nil {
			http.Error(w, fmt.Sprintf(`{"error":%q}`, err.Error()), http.StatusTooManyRequests)
			return
		}

		msgs := []provider.Message{
			{
				Role:    "system",
				Content: `Extract the refund decision as strict JSON with exact keys: {"order_id": string, "verdict": "APPROVE"|"DENY"|"MANUAL_REVIEW", "approved_idr": number, "reason_code": string}. Do not include extra keys.`,
			},
			{Role: "user", Content: req.TicketText},
		}

		var decision RefundDecision
		var totalPrompt, totalComp int
		var usedModel string
		attempts := 0

		for attempts <= contract.MaxSchemaRepairRetries {
			attempts++
			fault := ""
			if attempts == 1 {
				fault = req.FaultInject
			}
			res, err := router.CompleteWithFailover(r.Context(), provider.ChatRequest{
				Model:          contract.PrimaryModel,
				Messages:       msgs,
				Temperature:    0.0,
				MaxTokens:      160,
				ResponseSchema: "refund_decision",
				FaultInject:    fault,
			})
			if err != nil {
				http.Error(w, fmt.Sprintf(`{"error":%q}`, err.Error()), http.StatusBadGateway)
				return
			}
			usedModel = res.Model
			totalPrompt += res.PromptTokens
			totalComp += res.CompletionTokens

			if res.FinishReason == "length" {
				http.Error(w, `{"error":"structured output truncated by max_tokens"}`, http.StatusUnprocessableEntity)
				return
			}

			dec := json.NewDecoder(bytes.NewReader([]byte(res.Content)))
			dec.DisallowUnknownFields()
			parseErr := dec.Decode(&decision)
			if parseErr == nil && req.MaxRefundableIDR > 0 && decision.ApprovedIDR > req.MaxRefundableIDR {
				parseErr = fmt.Errorf("approved_idr %.0f exceeds max_refundable_idr %.0f", decision.ApprovedIDR, req.MaxRefundableIDR)
			}
			if parseErr == nil {
				cost := contract.ComputeCostUSD(usedModel, totalPrompt, totalComp)
				ledgerMgr.Record(req.TenantID, totalPrompt, totalComp, cost, false, false, res.FailedOver)
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(map[string]any{
					"decision":   decision,
					"attempts":   attempts,
					"model":      usedModel,
					"provider":   res.Provider,
					"cost_usd":   cost,
					"latency_us": time.Since(start).Microseconds(),
				})
				return
			}

			msgs = append(msgs,
				provider.Message{Role: "assistant", Content: res.Content},
				provider.Message{Role: "user", Content: "Repair invalid JSON so it strictly matches the required JSON schema: " + parseErr.Error()},
			)
		}

		http.Error(w, `{"error":"schema validation failed after repair retry"}`, http.StatusUnprocessableEntity)
	})

	return mux
}

func envOr(k, fallback string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return fallback
}

type upstreamEndpoint struct {
	name, url, key, model string
}

func resolvePrimaryUpstream() (upstreamEndpoint, bool) {
	switch {
	case os.Getenv("OPENAI_API_KEY") != "":
		return upstreamEndpoint{"openai", envOr("OPENAI_BASE_URL", "https://api.openai.com/v1"), os.Getenv("OPENAI_API_KEY"), os.Getenv("PRIMARY_MODEL")}, true
	case os.Getenv("GROQ_API_KEY") != "":
		return upstreamEndpoint{"groq", "https://api.groq.com/openai/v1", os.Getenv("GROQ_API_KEY"), envOr("PRIMARY_MODEL", "openai/gpt-oss-20b")}, true
	case os.Getenv("OPENCODE_API_KEY") != "" && os.Getenv("OPENCODE_BASE_URL") != "":
		return upstreamEndpoint{"opencode", os.Getenv("OPENCODE_BASE_URL"), os.Getenv("OPENCODE_API_KEY"), os.Getenv("PRIMARY_MODEL")}, true
	case os.Getenv("NVIDIA_API_KEY") != "":
		return upstreamEndpoint{"nvidia", "https://integrate.api.nvidia.com/v1", os.Getenv("NVIDIA_API_KEY"), envOr("PRIMARY_MODEL", "meta/llama-3.1-8b-instruct")}, true
	default:
		return upstreamEndpoint{}, false
	}
}

func resolveFallbackUpstream() (upstreamEndpoint, bool) {
	googleKey := envOr("GOOGLE_API_KEY", os.Getenv("GEMINI_API_KEY"))
	switch {
	case googleKey != "":
		return upstreamEndpoint{"gemini", envOr("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"), googleKey, envOr("FALLBACK_MODEL", "gemini-3.8-flash")}, true
	case os.Getenv("NVIDIA_API_KEY") != "":
		return upstreamEndpoint{"nvidia", "https://integrate.api.nvidia.com/v1", os.Getenv("NVIDIA_API_KEY"), envOr("FALLBACK_MODEL", "meta/llama-3.1-8b-instruct")}, true
	default:
		return upstreamEndpoint{}, false
	}
}

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	var mockStats provider.MockUpstreamStats

	pri, okPri := resolvePrimaryUpstream()
	if !okPri {
		srv := httptest.NewServer(provider.NewMockUpstreamHandler("openai", 18*time.Millisecond, 10*time.Millisecond, &mockStats))
		defer srv.Close()
		pri = upstreamEndpoint{name: "mock_openai", url: srv.URL}
	}

	fb, okFb := resolveFallbackUpstream()
	if !okFb {
		srv := httptest.NewServer(provider.NewMockUpstreamHandler("gemini", 12*time.Millisecond, 6*time.Millisecond, &mockStats))
		defer srv.Close()
		fb = upstreamEndpoint{name: "mock_gemini", url: srv.URL}
	}

	httpClient := &http.Client{Timeout: 30 * time.Second}
	var secondaryClient provider.Client = &provider.OpenAIHTTPClient{
		ProviderName:  fb.name,
		BaseURL:       fb.url,
		APIKey:        fb.key,
		ModelOverride: fb.model,
		HTTPClient:    httpClient,
	}
	if fb.name == "gemini" && fb.key != "" && os.Getenv("GEMINI_BASE_URL") == "" {
		secondaryClient = &provider.GeminiNativeClient{
			APIKey:        fb.key,
			ModelOverride: fb.model,
			HTTPClient:    httpClient,
		}
	}

	router := &provider.Router{
		Primary: &provider.OpenAIHTTPClient{
			ProviderName:  pri.name,
			BaseURL:       pri.url,
			APIKey:        pri.key,
			ModelOverride: pri.model,
			HTTPClient:    httpClient,
		},
		Secondary: secondaryClient,
	}

	cacheStore := cache.NewStore(contract.SemanticSimilarityThreshold)
	ledgerMgr := ledger.NewManager(contract.DefaultRPM, contract.DefaultDailyBudgetUSD)
	mux := NewGatewayMux(router, cacheStore, ledgerMgr)

	addr := ":" + envOr("PORT", "8080")
	srv := &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go func() {
		logger.Info("starting go-llm-gateway",
			"addr", addr,
			"schema_hash", contract.SchemaHash,
			"primary_provider", pri.name,
			"primary_model", envOr("PRIMARY_MODEL", pri.model),
			"fallback_provider", fb.name,
			"fallback_model", envOr("FALLBACK_MODEL", fb.model),
		)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("server error", "error", err)
			os.Exit(1)
		}
	}()

	<-ctx.Done()
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = srv.Shutdown(shutdownCtx)
}
