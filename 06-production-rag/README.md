# 06 — Production RAG Service

A production-grade, multi-tenant **Hybrid RAG** service pairing an offline Python evaluation plane (`ml/`) with a pure-Go online serving plane (`go/`), locked together by `feature_contract.json` (`v1.0.0`, schema hash `dd207803af0b`).

## What This Module Solves

1. **Exact-token blindness:** Dense embeddings blur exact policy codes (`POL-ORDE-01`), SKUs, and error strings where lexical BM25 excels.
2. **Cross-tenant leakage:** Post-filtering after global vector search risks returning unauthorized chunks or starving `top_k` results.
3. **Ungrounded hallucinations:** Answering when retrieval confidence is low or fabricating citations that never appeared in retrieved context.

This service combines **parallel `errgroup` BM25 + Vector search fused via Reciprocal Rank Fusion (`k=60`)**, **strict `tenant_id` pre-filtering**, **score-based abstention**, and **regex citation verification**.

---

## Architecture

```text
Offline Evaluation Plane (Python)              Online Serving Plane (Go :8086)
─────────────────────────────────              ───────────────────────────────
Kaggle Bitext Dataset                          POST /v1/rag/query | /v1/rag/ingest
        │                                                    │
   ml/train.py ──► artifacts/corpus.jsonl ───────────► internal/rag/engine.go
        │          artifacts/golden_fixtures.json       ├── errgroup: BM25 + Vector (256-dim)
        │          artifacts/benchmark_summary.json     ├── RRF Fusion (k=60)
        │                                               ├── Abstention (< 0.012)
        └──► feature_contract.json (dd207803af0b) ◄─────└── Citation Verifier ([POL-XXXX-NN#0])
                    │                                         │          │            │
                    ▼                                         ▼          ▼            ▼
          internal/contract/gen.go                     pgvector:pg19   RustFS     Grafana LGTM
                                                      (HNSW + GIN)    (S3 :9000)  (OTLP :4318)
```

---

## Offline Retrieval Benchmark

Evaluated on `bitext/bitext-gen-ai-chatbot-customer-support-dataset` (**27 canonical policy chunks across 11 tenants**, **135 held-out queries**):

| Strategy | Recall@5 | MRR@5 | Exact-Code R@5 | Paraphrase R@5 |
| :--- | :---: | :---: | :---: | :---: |
| `bm25_only` | `0.9185` | `0.7881` | `1.0000` | `0.8981` |
| `vector_only` | `0.9852` | `0.7898` | `0.9630` | `0.9907` |
| **`hybrid_rrf` (`k=60`)** | **`0.9926`** | **`0.8419`** | **`1.0000`** | **`0.9907`** |

![BM25 vs. Vector vs. Hybrid RRF](artifacts/rag_benchmark_summary.png)
![Canonical Policy Chunks per Tenant](artifacts/rag_tenant_distribution.png)
![Top-1 Hybrid RRF Score Distribution](artifacts/rag_rrf_score_distribution.png)

---

## Project Layout

```text
06-production-rag/
├── README.md
├── feature_contract.json                 # Shared Python/Go contract (dd207803af0b)
├── artifacts/
│   ├── corpus.jsonl                      # 27 canonical multi-tenant policy chunks
│   ├── benchmark_summary.json            # BM25 vs. Vector vs. Hybrid RRF metrics
│   ├── golden_fixtures.json              # 12 deterministic Python <-> Go parity cases
│   ├── rag_benchmark_summary.png
│   ├── rag_tenant_distribution.png
│   └── rag_rrf_score_distribution.png
├── ml/
│   ├── .python-version                   # Python 3.12
│   ├── pyproject.toml & uv.lock
│   ├── train.py                          # Kaggle ingestion, benchmark & fixture export
│   └── notebooks/
│       └── 06_hybrid_retrieval_audit.ipynb
└── go/
    ├── go.mod & go.sum
    ├── parity_test.go                    # Verifies codegen hash, RRF scores & isolation
    ├── cmd/server/main.go                # HTTP API (:8086), OTel middleware & synthesis
    └── internal/
        ├── contract/                     # go:generate codegen (doc.go, gen.go, contract_gen.go)
        ├── rag/engine.go                 # Parallel errgroup BM25 + Vector + RRF + grounding
        ├── store/store.go                # pgvector (pgx/v5) & RustFS (aws-sdk-go-v2)
        └── telemetry/otel.go             # OTLP/HTTP traces, metrics & slog->Loki bridge
```

---

## Quick Start

```bash
# From repository root:
make infra-up
make train MOD=06-production-rag
make notebook MOD=06-production-rag
make generate MOD=06-production-rag
make test MOD=06-production-rag

# Start Go RAG server wired to pgvector, RustFS, and LGTM:
(cd 06-production-rag/go && \
  DATABASE_URL="postgres://apprentice:apprentice_dev@localhost:5432/rag?sslmode=disable" \
  RUSTFS_ENDPOINT="http://localhost:9000" \
  go run ./cmd/server)
```