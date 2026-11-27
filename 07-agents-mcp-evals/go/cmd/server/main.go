package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"syscall"
	"time"

	"github.com/gunawanwijaya/apprentice/07-agents-mcp-evals/go/internal/agent"
	"github.com/gunawanwijaya/apprentice/07-agents-mcp-evals/go/internal/contract"
	"github.com/gunawanwijaya/apprentice/07-agents-mcp-evals/go/internal/mcp"
	"github.com/gunawanwijaya/apprentice/infra/telemetry"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	shutdownOTel, err := telemetry.Init(ctx, "07-agents-mcp-evals")
	if err != nil {
		slog.Warn("otel init failed, continuing without export", "err", err)
	} else {
		defer shutdownOTel(context.Background())
	}

	root := resolveModuleRoot()
	runtimeHash, err := computeContractHash(filepath.Join(root, "feature_contract.json"))
	if err != nil {
		slog.Error("load feature_contract.json", "err", err)
		os.Exit(1)
	}
	if runtimeHash != contract.SchemaHash {
		slog.Error("schema hash mismatch", "runtime", runtimeHash, "codegen", contract.SchemaHash)
		os.Exit(1)
	}

	orders, err := mcp.LoadOrdersSeed(filepath.Join(root, "artifacts", "orders_seed.json"))
	if err != nil {
		slog.Error("load orders_seed.json", "err", err)
		os.Exit(1)
	}
	policyMap := mcp.LoadPolicyFallbackMap(filepath.Join(root, "..", "06-production-rag", "artifacts", "corpus.jsonl"))
	mcpServer := mcp.NewServer(orders, policyMap, os.Getenv("RAG_SERVICE_URL"))
	ag := agent.New(mcpServer)

	meter := otel.Meter("07-agents-mcp-evals")
	runCounter, _ := meter.Int64Counter(
		"agent.runs",
		metric.WithDescription("Total agent runs partitioned by tenant, disposition, injection block, and HITL escalation"),
	)
	stepsHist, _ := meter.Int64Histogram(
		"agent.steps",
		metric.WithDescription("ReAct loop steps per agent invocation (max_steps = 4)"),
		metric.WithExplicitBucketBoundaries(1, 2, 3, 4, 5),
	)

	mux := http.NewServeMux()

	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{
			"status":       "ok",
			"schema_hash":  contract.SchemaHash,
			"orders_count": mcpServer.OrdersCount(),
			"mcp_version":  contract.MCPProtocolVersion,
		})
	})

	mux.Handle("POST /mcp", mcpServer)

	mux.HandleFunc("POST /v1/agent/run", func(w http.ResponseWriter, r *http.Request) {
		var req agent.RunRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}
		if req.SessionTenant == "" {
			req.SessionTenant = r.Header.Get("X-Tenant-Id")
		}
		res, err := ag.Run(r.Context(), req)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}

		runCounter.Add(r.Context(), 1, metric.WithAttributes(
			attribute.String("tenant_id", res.SessionTenant),
			attribute.String("disposition", res.Disposition),
			attribute.String("mutating_blocked", strconv.FormatBool(res.MutatingBlocked)),
			attribute.String("hitl_escalated", strconv.FormatBool(res.HITLEscalated)),
		))
		stepsHist.Record(r.Context(), int64(res.Steps), metric.WithAttributes(
			attribute.String("disposition", res.Disposition),
		))
		writeJSON(w, http.StatusOK, res)
	})

	addr := envOr("AGENT_ADDR", ":8087")
	srv := &http.Server{
		Addr:              addr,
		Handler:           telemetry.Middleware("07-agents-mcp-evals", mux),
		ReadHeaderTimeout: 5 * time.Second,
	}

	go func() {
		slog.Info("07-agents-mcp-evals listening", "addr", addr, "schema_hash", contract.SchemaHash, "orders", mcpServer.OrdersCount())
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			slog.Error("server error", "err", err)
		}
	}()

	<-ctx.Done()
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = srv.Shutdown(shutdownCtx)
}

func computeContractHash(path string) (string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	var generic any
	if err := json.Unmarshal(raw, &generic); err != nil {
		return "", err
	}
	canon, err := json.Marshal(generic)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(canon)
	return hex.EncodeToString(sum[:])[:12], nil
}

func resolveModuleRoot() string {
	if _, err := os.Stat("feature_contract.json"); err == nil {
		return "."
	}
	return ".."
}

func envOr(k, fallback string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return fallback
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
