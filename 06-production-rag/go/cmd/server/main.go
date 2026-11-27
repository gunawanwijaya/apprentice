package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/gunawanwijaya/apprentice/06-production-rag/go/internal/contract"
	"github.com/gunawanwijaya/apprentice/06-production-rag/go/internal/rag"
	"github.com/gunawanwijaya/apprentice/06-production-rag/go/internal/store"
	"github.com/gunawanwijaya/apprentice/infra/telemetry"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/propagation"
)

type queryReq struct {
	TenantID string `json:"tenant_id"`
	Query    string `json:"query"`
}

type queryResp struct {
	TenantID         string            `json:"tenant_id"`
	SchemaHash       string            `json:"schema_hash"`
	Abstained        bool              `json:"abstained"`
	Answer           string            `json:"answer"`
	Citations        []string          `json:"citations"`
	CitationVerified bool              `json:"citation_verified"`
	TopChunks        []rag.ScoredChunk `json:"top_chunks"`
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	shutdownOTel, err := telemetry.Init(ctx, "06-production-rag")
	if err != nil {
		slog.Warn("otel init failed, continuing without export", "err", err)
	} else {
		defer shutdownOTel(context.Background())
	}

	root := resolveModuleRoot()
	c, hash, err := rag.LoadContract(filepath.Join(root, "feature_contract.json"))
	if err != nil {
		slog.Error("load contract", "err", err)
		os.Exit(1)
	}
	if hash != contract.SchemaHash {
		slog.Error("contract hash mismatch with generated code", "runtime", hash, "codegen", contract.SchemaHash)
		os.Exit(1)
	}

	docs, err := rag.LoadCorpusJSONL(filepath.Join(root, "artifacts", "corpus.jsonl"))
	if err != nil {
		slog.Error("load corpus", "err", err)
		os.Exit(1)
	}
	engine := rag.NewEngine(c, hash, docs)

	meter := otel.Meter("06-production-rag")
	queryCounter, _ := meter.Int64Counter(
		"rag.queries",
		metric.WithDescription("Total RAG queries partitioned by tenant, abstention, and citation verification"),
	)
	ingestCounter, _ := meter.Int64Counter(
		"rag.ingests",
		metric.WithDescription("Total RAG chunks ingested partitioned by tenant"),
	)
	rrfScoreHist, _ := meter.Float64Histogram(
		"rag.top_rrf_score",
		metric.WithDescription("Top-1 Hybrid RRF score distribution (abstention threshold = 0.012)"),
		metric.WithExplicitBucketBoundaries(0.0, 0.005, 0.010, 0.012, 0.015, 0.0165, 0.020, 0.025, 0.030, 0.032, 0.033, 0.035),
	)

	var pgStore *store.PGStore
	if dsn := os.Getenv("DATABASE_URL"); dsn != "" {
		pg, err := store.NewPGStore(ctx, dsn, c.Retrieval.EmbedDim)
		if err != nil {
			slog.Warn("pgvector disabled", "err", err)
		} else {
			defer pg.Close()
			pgStore = pg
			for _, d := range docs {
				_ = pgStore.UpsertChunk(ctx, d)
			}
			slog.Info("pgvector synced", "chunks", len(docs))
		}
	}

	var s3Store *store.S3Store
	if ep := os.Getenv("RUSTFS_ENDPOINT"); ep != "" {
		ak := envOr("RUSTFS_ACCESS_KEY", "apprentice")
		sk := envOr("RUSTFS_SECRET_KEY", "apprentice_secret_123")
		s3s, err := store.NewS3Store(ctx, ep, ak, sk, envOr("RUSTFS_BUCKET", "rag-docs"))
		if err != nil {
			slog.Warn("rustfs disabled", "err", err)
		} else {
			s3Store = s3s
			slog.Info("rustfs connected", "endpoint", ep)
		}
	}

	gatewayURL := os.Getenv("LLM_GATEWAY_URL")
	mux := http.NewServeMux()

	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{
			"status":      "ok",
			"schema_hash": engine.SchemaHash,
			"corpus_size": engine.CorpusSize(),
		})
	})

	mux.HandleFunc("POST /v1/rag/ingest", func(w http.ResponseWriter, r *http.Request) {
		var ch rag.Chunk
		if err := json.NewDecoder(r.Body).Decode(&ch); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}
		if ch.ChunkID == "" || ch.TenantID == "" || ch.Content == "" {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "chunk_id, tenant_id, and content are required"})
			return
		}
		ch.TenantID = strings.ToUpper(strings.TrimSpace(ch.TenantID))
		engine.UpsertChunk(ch)
		ingestCounter.Add(r.Context(), 1, metric.WithAttributes(attribute.String("tenant_id", ch.TenantID)))

		if pgStore != nil {
			if err := pgStore.UpsertChunk(r.Context(), ch); err != nil {
				slog.WarnContext(r.Context(), "pgvector upsert failed", "chunk_id", ch.ChunkID, "err", err)
			}
		}
		if s3Store != nil {
			if err := s3Store.ArchiveChunk(r.Context(), ch); err != nil {
				slog.WarnContext(r.Context(), "rustfs archive failed", "chunk_id", ch.ChunkID, "err", err)
			}
		}
		writeJSON(w, http.StatusCreated, map[string]any{
			"status":      "ingested",
			"chunk_id":    ch.ChunkID,
			"corpus_size": engine.CorpusSize(),
		})
	})

	mux.HandleFunc("POST /v1/rag/query", func(w http.ResponseWriter, r *http.Request) {
		var req queryReq
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}
		tenant := strings.ToUpper(strings.TrimSpace(req.TenantID))
		hits, err := engine.Retrieve(r.Context(), req.Query, tenant)
		if err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
			return
		}

		topScore := 0.0
		if len(hits) > 0 {
			topScore = hits[0].Score
		}
		rrfScoreHist.Record(r.Context(), topScore, metric.WithAttributes(attribute.String("tenant_id", tenant)))

		maxCtx := engine.Contract.Retrieval.FinalContextChunks
		if len(hits) > maxCtx {
			hits = hits[:maxCtx]
		}
		if engine.ShouldAbstain(hits) {
			queryCounter.Add(r.Context(), 1, metric.WithAttributes(
				attribute.String("tenant_id", tenant),
				attribute.String("abstained", "true"),
				attribute.String("citation_verified", "false"),
			))
			writeJSON(w, http.StatusOK, queryResp{
				TenantID:   tenant,
				SchemaHash: engine.SchemaHash,
				Abstained:  true,
				Answer:     "I cannot answer this reliably from the authorized policy documents for this tenant.",
				TopChunks:  hits,
			})
			return
		}

		answer := synthesizeAnswer(r.Context(), gatewayURL, req.Query, hits)
		verified, cited, _ := rag.VerifyCitations(answer, hits)
		queryCounter.Add(r.Context(), 1, metric.WithAttributes(
			attribute.String("tenant_id", tenant),
			attribute.String("abstained", "false"),
			attribute.String("citation_verified", strconv.FormatBool(verified)),
		))
		writeJSON(w, http.StatusOK, queryResp{
			TenantID:         tenant,
			SchemaHash:       engine.SchemaHash,
			Abstained:        false,
			Answer:           answer,
			Citations:        cited,
			CitationVerified: verified,
			TopChunks:        hits,
		})
	})

	addr := envOr("RAG_ADDR", ":8086")
	srv := &http.Server{
		Addr:              addr,
		Handler:           telemetry.Middleware("06-production-rag", mux),
		ReadHeaderTimeout: 5 * time.Second,
	}
	go func() {
		slog.Info("06-production-rag listening", "addr", addr, "schema_hash", engine.SchemaHash, "chunks", engine.CorpusSize())
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			slog.Error("server error", "err", err)
		}
	}()

	<-ctx.Done()
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = srv.Shutdown(shutdownCtx)
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

func synthesizeAnswer(ctx context.Context, gatewayURL, query string, hits []rag.ScoredChunk) string {
	if gatewayURL != "" {
		var sb strings.Builder
		sb.WriteString("Answer using ONLY the policy context below and cite chunk IDs in [CHUNK_ID] format:\n")
		for _, h := range hits {
			fmt.Fprintf(&sb, "[%s] %s\n%s\n\n", h.Chunk.ChunkID, h.Chunk.Header, h.Chunk.Content)
		}
		payload, _ := json.Marshal(map[string]any{
			"messages": []map[string]string{
				{"role": "system", "content": sb.String()},
				{"role": "user", "content": query},
			},
		})
		req, _ := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(gatewayURL, "/")+"/v1/chat/completions", bytes.NewReader(payload))
		otel.GetTextMapPropagator().Inject(ctx, propagation.HeaderCarrier(req.Header))
		req.Header.Set("Content-Type", "application/json")
		resp, err := (&http.Client{Timeout: 10 * time.Second}).Do(req)
		if err == nil {
			defer resp.Body.Close()
			var out struct {
				Choices []struct {
					Message struct{ Content string } `json:"message"`
				} `json:"choices"`
			}
			if json.NewDecoder(resp.Body).Decode(&out) == nil && len(out.Choices) > 0 && out.Choices[0].Message.Content != "" {
				return out.Choices[0].Message.Content
			}
		}
	}
	top := hits[0].Chunk
	return fmt.Sprintf("Based on policy %s [%s]: %s", top.PolicyCode, top.ChunkID, top.Content)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
