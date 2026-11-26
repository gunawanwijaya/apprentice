# 06 — Production RAG Service

A production-grade, multi-tenant **Hybrid RAG** service pairing an offline Python evaluation plane (`ml/`) with a pure-Go online serving plane (`go/`), locked together by a deterministic `feature_contract.json` (`v1.0.0`, schema hash `dd207803af0b`).

## What This Module Solves

Naive vector-only RAG fails in production on three predictable failure modes:
1. **Exact-token blindness:** Dense embeddings blur exact policy codes (`POL-ORDE-01`), SKUs, and error strings where lexical BM25 excels.
2. **Cross-tenant leakage:** Post-filtering after global vector search risks returning unauthorized chunks or starving `top_k` results.
3. **Ungrounded hallucinations:** Answering when retrieval confidence is low or fabricating citations that never appeared in retrieved context.

This service addresses all three with **parallel BM25 + Vector search fused via Reciprocal Rank Fusion (`k=60`)**, **strict `tenant_id` pre-filtering**, **score-based abstention**, and **regex citation verification**.

---

## Architecture

```text
Offline Evaluation Plane (Python)              Online Serving Plane (Go :8086)
─────────────────────────────────              ───────────────────────────────
Kaggle Bitext Dataset                          POST /v1/rag/query | /v1/rag/ingest
        │                                                    │
   ml/train.py ──► artifacts/corpus.jsonl ───────────► internal/rag/engine.go
        │          artifacts/golden_fixtures.json       ├── errgroup: BM25 + Vector (256-dim)
        │          artifacts/eval_summary.json          ├── RRF Fusion (k=60)
        │                                               ├── Abstention (< 0.012)
        └──► feature_contract.json (dd207803af0b) ◄─────└── Citation Verifier ([POL-XXXX-NN#0])
                                                              │          │            │
                                                              ▼          ▼            ▼
                                                       pgvector:pg19   RustFS     Grafana LGTM
                                                      (HNSW + GIN)    (S3 :9000)  (OTLP :4318)
                                                                                       │
                                                                          Optional: 05-llm-gateway
```

---

## Offline Retrieval Benchmark

Evaluated on the Kaggle `bitext/bitext-gen-ai-chatbot-customer-support-dataset` grouped into **27 canonical policy chunks across 11 tenants** with **135 held-out queries** (real customer paraphrases + exact policy-code lookups):

| Strategy | Recall@5 | MRR@5 | Exact-Code R@5 | Paraphrase R@5 |
| :--- | :---: | :---: | :---: | :---: |
| `bm25_only` | `0.9185` | `0.7881` | `1.0000` | `0.8981` |
| `vector_only` | `0.9852` | `0.7898` | `0.9630` | `0.9907` |
| **`hybrid_rrf` (`k=60`)** | **`0.9926`** | **`0.8419`** | **`1.0000`** | **`0.9907`** |

![Canonical Policy Chunks per Tenant](artifacts/plots/01_tenant_distribution.png)
![BM25 vs. Vector vs. Hybrid RRF](artifacts/plots/02_retrieval_strategy_comparison.png)
![Top-1 Hybrid RRF Score Distribution](artifacts/plots/03_rrf_score_distribution.png)

**Key takeaway:** `bm25_only` achieves `1.0000` on exact policy codes but drops to `0.8981` on paraphrases; `vector_only` hits `0.9907` on paraphrases but misses exact codes (`0.9630`). Combining both via RRF (`k=60`) reaches **`0.9926` Recall@5** and boosts **MRR@5 by +5.2 points** (`0.8419`).

---

## Python ↔ Go Parity Contract

Both `ml/train.py` and `go/internal/rag/engine.go` enforce identical math and contract hashing (`feature_contract.json`):

- **Schema Hash (`dd207803af0b`):** Canonical JSON SHA-256 prefix verified at startup and in `parity_test.go`.
- **Tokenization:** Lowercase regex normalization (`[^a-z0-9\-_/\s]+`) + shared English stop-word list.
- **Embeddings (`256`-dim):** Deterministic FNV-1a 32-bit feature hashing over stemmed unigrams (`w:`), bigrams (`b:`), and character 3-grams (`c:`), L2-normalized—zero CGO required.
- **BM25:** `k1 = 1.5`, `b = 0.75`, IDF smoothing $\ln(1 + \frac{N - df + 0.5}{df + 0.5})$.
- **RRF Fusion:** $\text{score}(d) = \sum_{r \in \{\text{bm25}, \text{vec}\}} \frac{1}{60 + \text{rank}_r(d)}$.
- **Grounding:** Fails closed when `tenant_id` is empty, abstains when top RRF score `< 0.012`, and validates every `[POL-XXXX-NN#0]` citation against the retrieved chunks.

---

## Persistence & Observability

- **PostgreSQL (`pgvector:pg19`):** Auto-migrates `rag_chunks` with `vector(256)` HNSW cosine index (`idx_rag_chunks_hnsw`), generated English `tsvector` GIN index (`idx_rag_chunks_tsv`), and `tenant_id` B-tree index. Syncs corpus on startup and persists live ingests.
- **RustFS (`:9000` / `:9001`):** S3-compatible object store archiving every ingested chunk to `rag-docs/tenants/<TENANT>/<CHUNK_ID>.json` via `aws-sdk-go-v2` (path-style).
- **Grafana LGTM (`:3000` / `:4318`):** Exports distributed traces (`POST /v1/rag/query` → `rag.retrieve` → `rag.bm25` / `rag.vector`) and `http.server.request.duration` histograms via OTLP/HTTP to the provisioned **Apprentice Stack** dashboard.

---

## Project Layout

```text
06-production-rag/
├── feature_contract.json            # Shared Python/Go retrieval & grounding contract
├── artifacts/
│   ├── corpus.jsonl                 # 27 canonical multi-tenant policy chunks
│   ├── eval_summary.json            # BM25 vs. Vector vs. Hybrid RRF metrics
│   └── golden_fixtures.json         # 12 deterministic Python <-> Go parity test cases
├── ml/
│   ├── train.py                     # Kaggle ingestion, benchmark runner, artifact exporter
│   └── hybrid_retrieval_audit.ipynb # Interactive charts, RRF k sweep & score histograms
└── go/
    ├── cmd/server/main.go           # HTTP API (:8086), OTel middleware, gateway synthesis
    └── internal/
        ├── rag/
        │   ├── engine.go            # Parallel errgroup BM25 + Vector + RRF + citation check
        │   └── parity_test.go       # Exact Python <-> Go score parity & tenant isolation tests
        ├── store/store.go           # pgvector (pgx/v5) & RustFS (aws-sdk-go-v2) adapters
        └── telemetry/otel.go        # OTLP/HTTP trace & metric exporters
```

---

## Quick Start

### 1. Start Local Infrastructure (Repo Root)

```bash
make infra-up
```

### 2. Run Offline Benchmark & Export Artifacts

```bash
make train MOD=06-production-rag
```

### 3. Run Python ↔ Go Parity & Isolation Tests

```bash
(cd 06-production-rag/go && go test -v ./...)
```

### 4. Start the Go RAG Server

```bash
(cd 06-production-rag/go && \
  DATABASE_URL="postgres://apprentice:apprentice_dev@localhost:5432/rag?sslmode=disable" \
  RUSTFS_ENDPOINT="http://localhost:9000" \
  go run ./cmd/server)
```

Optional environment variables:
- `RAG_ADDR` (default `:8086`)
- `OTEL_EXPORTER_OTLP_ENDPOINT` (default `http://localhost:4318`)
- `LLM_GATEWAY_URL` (e.g. `http://localhost:8085` to synthesize via `05-llm-gateway`; falls back to deterministic grounded synthesis when unset)
- `RUSTFS_ACCESS_KEY`, `RUSTFS_SECRET_KEY`, `RUSTFS_BUCKET` (default `rag-docs`)

---

## API Reference

### Health Check

```bash
curl -s http://localhost:8086/healthz | jq .
```

### Live Policy Ingest (Updates Memory + `pgvector` + RustFS)

```bash
curl -s http://localhost:8086/v1/rag/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "chunk_id": "POL-ORDE-99#0",
    "tenant_id": "ORDER",
    "policy_code": "POL-ORDE-99",
    "intent": "rush_cancel",
    "header": "[Tenant: ORDER | Policy: POL-ORDE-99 | Topic: rush cancel]",
    "content": "Rush orders can be canceled within 15 minutes of checkout."
  }' | jq .
```

### Multi-Tenant Hybrid Query

```bash
curl -s http://localhost:8086/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"ORDER","query":"Can I cancel a rush order?"}' | jq .
```