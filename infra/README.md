# Shared Infrastructure & Unified Observability (`infra/`)

Provides the local storage, object store, shared Go OpenTelemetry SDK (`infra/telemetry`), provisioned Grafana dashboards (`00` Home + `01`–`07`), and multi-service golden-fixture traffic replayer (`infra/replay.py`) for the `apprentice` monorepo.

## Components & Port Map

| Component | Port(s) | Purpose |
| :--- | :--- | :--- |
| **`pgvector/pgvector:pg19`** | `:5432` | PostgreSQL 19 with `vector` HNSW + `tsvector` GIN hybrid indexing (`06-production-rag`) |
| **`rustfs/rustfs`** | `:9000` / `:9001` | S3-compatible document & artifact archive pushing native metrics/logs over OTLP |
| **`grafana/otel-lgtm`** | `:3000`, `:4317`, `:4318` | Grafana UI (`admin`/`admin`), OTel Collector, Prometheus, Tempo, and Loki |
| **`01` → `07` Go AI Fleet** | `:8081` → `:8087` | Dedicated non-colliding HTTP ports across all 7 modules (`make serve-all`) |

## Directory Layout

```text
infra/
├── README.md
├── replay.py                             # Streams golden_fixtures.json across :8081 -> :8087
├── telemetry/                            # Shared Go OTel SDK (traces, sub-ms histograms, JSON slog -> Loki)
│   ├── go.mod & go.sum
│   └── otel.go
└── lgtm/
    ├── otelcol-config.yaml               # OTLP gRPC/HTTP + PostgreSQL receiver + LGTM self-scrape
    ├── dashboards-provisioning.yaml      # Auto-loads all dashboards every 10s into "Apprentice Infra"
    └── dashboards/
        ├── apprentice-stack.json         # 00 — Apprentice Home (LGTM, pgvector, RustFS & 01-07 Fleet)
        ├── 01-delivery-sla-breach.json   # Live breach probability & adaptive CQR interval width
        ├── 02-classic-nlp.json           # Live intent/queue/priority volume, PII mask % & BM25 scores
        ├── 03-transformer-onnx.json      # Live FP32 vs. INT8 ONNX latency & confidence comparison
        ├── 04-multimodal-ai.json         # Live FFmpeg demux vs. concurrent ONNX stage waterfall & dedup
        ├── 05-llm-gateway.json           # Live exact/semantic cache hit %, entity guard, failovers & USD
        ├── 06-production-rag.json        # Live tenant density, RRF score heatmap, abstention & citations
        └── 07-agents-mcp-evals.json      # Live ReAct dispositions, injection/HITL gates & MCP tool calls
```

## End-to-End Fleet Demo

```bash
# From repository root:
make infra-up        # Start pgvector, RustFS, and Grafana LGTM
make serve-all       # Build & launch all 7 Go servers (:8081 -> :8087)
make demo-traffic    # Replay golden fixtures across all 7 services (Ctrl+C to stop)
make stop-all        # Stop background Go servers cleanly
```

Open `http://localhost:3000` (`admin` / `admin`) to inspect **`00 — Apprentice Home`** and drill into any module via the **`Module Dashboards (01–07)`** dropdown, with full **Prometheus Exemplar (`✦`) → Tempo Trace → Loki JSON Log** correlation via `trace_id`.