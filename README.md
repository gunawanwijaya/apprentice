# Apprentice — Production ML & AI Systems in Python + Go

A hands-on monorepo of **6 production AI/ML systems** plus a **shared observability & storage stack**. Every module pairs an **offline Python training & evaluation plane (`ml/`)** with a **zero-CGO / pure-Go online serving plane (`go/`)**, bound together by a SHA-256 versioned **`feature_contract.json`** and deterministic **`golden_fixtures.json`** parity tests.

---

## Architecture Philosophy

```text
┌──────────────────────────────┐         ┌──────────────────────────────────┐
│ Offline ML Plane (Python/uv) │         │ Online Serving Plane (Go 1.23+)  │
│  • Real dataset ingestion    │         │  • Strict contract hash check    │
│  • Ablation & calibration    │         │  • Sub-millisecond concurrency   │
│  • Export ONNX / JSONL       ├────────►│  • Deterministic preprocessing   │
│  • Export golden_fixtures    │         │  • parity_test.go (1:1 w/ Python)│
└──────────────┬───────────────┘         └────────────────┬─────────────────┘
               │                                          │
               └──────────► feature_contract.json ◄───────┘
                                                          │
                         ┌────────────────────────────────┴────────────────┐
                         ▼                        ▼                        ▼
                  pgvector:pg19             RustFS S3 (:9000)       Grafana LGTM (:3000)
                 (HNSW + GIN :5432)        (Object Archive)         (OTLP :4317 / :4318)
```

---

## Module Matrix

| Module | Domain | Offline Plane (`ml/`) | Online Plane (`go/`) | Port | Docs |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **`01-delivery-sla-breach`** | Tabular ML & Conformal Prediction | GBDT SLA classifier + CQR (`p10`/`p50`/`p90`) ONNX | Pure-Go ONNX inference & calibrated ETA intervals | `:8081` | [README](01-delivery-sla-breach/README.md) |
| **`02-classic-nlp`** | Lexical & Sequence NLP | TF-IDF/Linear ONNX (`intent`/`priority`/`queue`), CRF NER, BM25 | Pure-Go Viterbi decoder, PII redactor & BM25 | `:8082` | [README](02-classic-nlp/README.md) |
| **`03-transformer-onnx`** | Quantized Transformer Serving | MiniLM fine-tuning, `FP32` vs. dynamic `INT8` ONNX benchmark | Pure-Go WordPiece tokenizer + quantized inference | `:8083` | [README](03-transformer-onnx/README.md) |
| **`04-multimodal-ai`** | Vision + Audio Multimodal | eKYC & unboxing CNNs + Audio VAD ONNX export | Pure-Go video demux, image/audio DSP & fusion | `:8084` | [README](04-multimodal-ai/README.md) |
| **`05-llm-gateway`** | LLM Routing, Cache & FinOps | Multi-provider cost/latency simulation & cache economics | Semantic cache, entity guardrails, budget ledger | `:8085` | [README](05-llm-gateway/README.md) |
| **`06-production-rag`** | Multi-Tenant Hybrid RAG | Kaggle Bitext BM25 vs. Vector vs. RRF (`k=60`) audit | `errgroup` Hybrid RRF, abstention, citation check, `pgvector` + RustFS + OTel | `:8086` | [README](06-production-rag/README.md) |
| **`infra/`** | Storage & LGTM Observability | Provisioned Grafana dashboards & OTel Collector pipelines | `pgvector:pg19` (`:5432`), RustFS (`:9000`), LGTM (`:3000`, `:4318`) | `:3000` | [README](infra/README.md) |

---

## Uniform Module Layout

Every `0X-<module>/` directory follows the exact same structure:

```text
0X-<module>/
├── README.md                        # Problem statement, architecture, benchmark plots & API docs
├── feature_contract.json            # Versioned Python <-> Go contract (verified via SHA-256 hash)
├── artifacts/                       # Exported models (.onnx / .jsonl), benchmark_summary.json,
│                                    # golden_fixtures.json, and benchmark .png plots
├── ml/
│   ├── .python-version              # Pinned Python 3.12 (managed by uv)
│   ├── pyproject.toml & uv.lock     # Reproducible offline dependencies
│   ├── train.py                     # Dataset pipeline, evaluation, and artifact/fixture exporter
│   └── notebooks/
│       └── 0X_<name>.ipynb          # Executed EDA & ablation notebook
└── go/
    ├── go.mod & go.sum              # Independent Go module
    ├── parity_test.go               # Verifies exact numerical & contract parity against Python
    ├── cmd/server/main.go           # HTTP server (:808X) with /healthz & graceful shutdown
    └── internal/
        └── contract/                # go:generate codegen binding feature_contract.json
```

---

## Unified Workflow (`Makefile`)

Run any module or the shared infrastructure from the repository root:

```bash
# 1. Start / stop shared backing services (pgvector, RustFS, Grafana LGTM)
make infra-up
make infra-down

# 2. Train & export artifacts + golden fixtures for a module
make train MOD=06-production-rag

# 3. Execute & update a module's Jupyter benchmark notebook in-place
make notebook MOD=06-production-rag

# 4. Run Go contract codegen and Python <-> Go parity tests
make generate MOD=06-production-rag
make test MOD=06-production-rag

# 5. Run parity tests across ALL modules (01 -> 06)
make test-all

# 6. Start a module's Go HTTP server
make serve MOD=06-production-rag
```