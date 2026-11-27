# Apprentice — Production ML & AI Systems in Python + Go

[![CI](https://github.com/gunawanwijaya/apprentice/actions/workflows/ci.yml/badge.svg)](https://github.com/gunawanwijaya/apprentice/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A hands-on monorepo of **7 production AI/ML systems** plus a **shared observability & storage stack**. Every module pairs an **offline Python training & evaluation plane (`ml/`)** with an **isolated Go online serving & control plane (`go/`, embedded ONNX Runtime via CGO in `01`–`04` and pure-Go `net/http` in `05`–`07`)**, bound together by a SHA-256 versioned **`feature_contract.json`**, deterministic **`golden_fixtures.json`** parity gates, and shared OpenTelemetry instrumentation (`infra/telemetry`).

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
| **`01-delivery-sla-breach`** | Tabular ML & Conformal Prediction | GBDT SLA classifier + CQR (`p10`/`p50`/`p90`) ONNX | Go ONNX inference (`*float32` `null`→`NaN`) & calibrated ETA intervals | `:8081` | [README](01-delivery-sla-breach/README.md) |
| **`02-classic-nlp`** | Lexical & Sequence NLP | TF-IDF/Linear ONNX (`intent`/`priority`/`queue`), CRF NER, BM25 | Pure-Go Viterbi decoder, PII redactor, BM25 & ONNX text triage | `:8082` | [README](02-classic-nlp/README.md) |
| **`03-transformer-onnx`** | Quantized Transformer Serving | 5-model text benchmark (`bert-tiny`), `FP32` vs. dynamic `INT8` ONNX | Pure-Go WordPiece tokenizer + quantized ONNX encoder | `:8083` | [README](03-transformer-onnx/README.md) |
| **`04-multimodal-ai`** | Vision + Audio Multimodal | eKYC & unboxing CNNs + Audio VAD ONNX export | `ffmpeg` demux, zero-OpenCV PNG/WAV DSP & `errgroup` ONNX fusion | `:8084` | [README](04-multimodal-ai/README.md) |
| **`05-llm-gateway`** | LLM Routing, Cache & FinOps | Multi-provider cost/latency simulation & cache economics | Pure-Go SSE proxy, entity-safe cache, 429 failover & budget ledger | `:8085` | [README](05-llm-gateway/README.md) |
| **`06-production-rag`** | Multi-Tenant Hybrid RAG | Kaggle Bitext BM25 vs. Vector vs. RRF (`k=60`) audit | `errgroup` Hybrid RRF, abstention floor, citation check, `pgvector` + RustFS | `:8086` | [README](06-production-rag/README.md) |
| **`07-agents-mcp-evals`** | Bounded Agents, MCP & Evals | Guardrail ablation (60 runs) & CoT Binary Judge calibration ($\kappa=0.92$) | Streamable HTTP JSON-RPC MCP server, bounded ReAct loop & HITL cap | `:8087` | [README](07-agents-mcp-evals/README.md) |
| **`infra/`** | Storage & LGTM Observability | 8 provisioned Grafana dashboards (`00` Home + `01`–`07`) & `replay.py` | Shared `infra/telemetry` SDK, `pgvector:pg19` (`:5432`), RustFS (`:9000`), LGTM (`:3000`) | `:3000` | [README](infra/README.md) |

---

## Tiered ROI & Architecture Decision Matrix

Solve high-volume deterministic and narrow tasks on sub-5ms CPU tiers before escalating to metered remote GPUs:

| Workload | Anti-Pattern | `apprentice` Tier | Typical p95 | Cost Envelope |
| :--- | :--- | :--- | :---: | :---: |
| **Tabular SQL Risk & ETA** | Prompting LLMs over JSON rows | **`01`** LightGBM + CQR ONNX | `0.5 – 6 ms` | `~$0.50 / 1M reqs` |
| **PII Masking & Exact Codes** | Cloud LLM redaction / vector-only search | **`02`** Regex/Checksum + CRF + BM25 | `0.15 – 3 ms` | `~$0.30 / 1M reqs` |
| **Nuanced Intent / Negation** | Autoregressive Decoder LLM | **`03`** `INT8` Encoder ONNX | `2 – 15 ms` | `~$1.50 / 1M reqs` |
| **KYC / Claim Video & Audio** | Raw 30 FPS video straight to VLM | **`04`** 1 FPS Dedup + CNN/VAD Gate | `3 – 45 ms` ONNX | `-60% to -85%` VLM spend |
| **LLM Extraction & Chat** | Uncached prose calls from app pods | **`05`** Entity-Safe Cache + Compact JSON | `0.4 ms` hit / `~730 ms` miss | `-65% to -95%` token spend |
| **Multi-Tenant Policy Q&A** | Fine-tuning weights on changing SOPs | **`06`** BM25 + `pgvector` RRF + Abstain | `<1 ms` retrieve | `$0` LLM spend on abstains |
| **Multi-System Dispute Triage** | Unbounded agent over 50 REST APIs | **`07`** Curated MCP + `max_steps=4` + HITL | `1.5 – 5 s` | Reserved for complex tier |

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
│   ├── pyproject.toml & uv.lock     # Reproducible, vulnerability-audited offline dependencies
│   ├── train.py                     # Dataset pipeline, evaluation, and artifact/fixture exporter
│   └── notebooks/
│       └── 0X_<name>.ipynb          # Executed EDA & ablation notebook
└── go/
    ├── go.mod & go.sum              # Standalone Go module (GOWORK=off reproducible)
    ├── parity_test.go               # Verifies exact numerical, contract & security parity against Python
    ├── cmd/server/main.go           # HTTP server (:8081–:8087) with /healthz, OTel & graceful shutdown
    └── internal/
        └── contract/                # go:generate codegen binding feature_contract.json
```

---

## Unified Workflow (`Makefile`)

Run any module or the full 7-service observability fleet from the repository root:

```bash
# 1. Start shared backing services, launch all 7 Go servers (:8081–:8087) & stream live traffic to Grafana (:3000)
make infra-up
make serve-all
make demo-traffic
make stop-all

# 2. Train & export artifacts + golden fixtures for a single module
make train MOD=07-agents-mcp-evals

# 3. Execute & update a module's Jupyter benchmark notebook in-place
make notebook MOD=07-agents-mcp-evals

# 4. Run Go contract codegen and non-cached Python <-> Go parity tests across ALL 7 modules
make generate-all
make test-all

# 5. Start a single module's Go HTTP server
make serve MOD=07-agents-mcp-evals
```