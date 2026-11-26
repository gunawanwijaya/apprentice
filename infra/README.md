# Apprentice Shared Infrastructure (`infra/`)

Local production-parity backing stack for the `apprentice` monorepo: **PostgreSQL 19 + `pgvector`** (hybrid vector/FTS storage), **RustFS** (S3-compatible object storage with native OTLP telemetry), and **Grafana `otel-lgtm`** (OpenTelemetry Collector, Loki, Grafana, Tempo, Prometheus, Pyroscope).

---

## Architecture & Telemetry Flow

```text
Go AI Services (:8085 05-llm-gateway, :8086 06-production-rag)
  │  ├── SQL / HNSW + GIN ──────────────► pgvector:pg19 (:5432)
  │  ├── S3 PutObject / GetObject ──────► rustfs (:9000 API / :9001 Console)
  │  └── OTLP/HTTP (traces, metrics) ───┐       │
  │                                     ▼       ▼ OTLP/HTTP (:4318)
  │                           ┌───────────────────────────────────┐
  │                           │ apprentice-lgtm (:3000, :4317/18) │
  │                           │  ├─ OTel Collector (v0.165+)      │
  │                           │  │   ├─ otlp receiver (:4317/18)  │
  │                           │  │   ├─ postgresql receiver ──────┼──► scrapes pgvector (:5432)
  │                           │  │   └─ prometheus self-scrape    │
  │                           │  ├─ Prometheus (:9090 internal)   │
  │                           │  ├─ Loki (:3100 internal)         │
  │                           │  ├─ Tempo (:3200 internal)        │
  │                           │  └─ Grafana (:3000 UI)            │
  │                           └───────────────────────────────────┘
```

---

## Service & Port Matrix

| Container | Image | Host Ports | Default Credentials / Config | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `apprentice-pgvector` | `pgvector/pgvector:pg19` | `5432` | `apprentice` / `apprentice_dev` (`db: rag`) | Vector (`HNSW`) + lexical (`tsvector` `GIN`) store |
| `apprentice-rustfs` | `rustfs/rustfs:latest` | `9000` (S3), `9001` (UI) | `apprentice` / `apprentice_secret_123` | S3 artifact & raw policy chunk archive |
| `apprentice-lgtm` | `grafana/otel-lgtm:latest` | `3000` (UI), `4317` (gRPC), `4318` (HTTP) | `admin` / `admin` (anonymous Admin enabled locally) | Unified OTel metrics, logs, traces & dashboards |

---

## Directory Layout

```text
infra/
├── README.md                            # Infrastructure & observability runbook
└── lgtm/
    ├── otelcol-config.yaml              # OTel Collector receivers, batch processor & LGTM exporters
    ├── dashboards-provisioning.yaml     # Auto-provisions JSON dashboards every 10s
    └── dashboards/
        └── apprentice-stack.json        # Default home dashboard (LGTM, pgvector, RustFS, Go services)
```

---

## Quick Start

Run from the **repository root**:

```bash
# Start all backing containers in the background
make infra-up

# Check container health
docker compose ps

# Tail logs across the stack
docker compose logs -f

# Stop stack (preserves named volumes)
make infra-down
```

Open **`http://localhost:3000`** to land directly on the **Apprentice Stack** home dashboard, or **`http://localhost:9001`** for the RustFS S3 console.

---

## Environment Overrides (`.env`)

`docker-compose.yml` works out-of-the-box with safe local defaults. Override any value via an untracked root `.env` file:

```bash
POSTGRES_USER=apprentice
POSTGRES_PASSWORD=apprentice_dev
POSTGRES_DB=rag
RUSTFS_ACCESS_KEY=apprentice
RUSTFS_SECRET_KEY=apprentice_secret_123
GF_ADMIN_PASSWORD=admin
```

---

## Provisioned Dashboard (`apprentice-stack.json`)

1. **Row 1 — LGTM Self-Monitoring:** Daemon `up` status (`grafana`, `loki`, `tempo`, `prometheus`, `otelcol-contrib`), OTel Collector accepted signals/sec, RSS memory, goroutines, and refused/failed signal rates.
2. **Row 2 — PostgreSQL (`pgvector`):** Active backends, transaction commits vs. rollbacks/sec, and database size (`MB`) via the OTel `postgresql` receiver.
3. **Row 3 — RustFS S3 & Loki Logs:** Native `rustfs_*` OTLP metrics pushed from RustFS and `{service_name=~".+"}` structured logs from Loki.
4. **Row 4 — Go AI Services (`05-llm-gateway` & `06-production-rag`):** Request rate by `service_name` + `http_route` and `p95` HTTP latency (`ms`) computed from `http_server_request_duration_seconds_bucket`.

---

## Troubleshooting & Operational Notes

- **PostgreSQL 18+ / `pg19` Volume Mount Path:** Official Postgres 18+ images store versioned data under `/var/lib/postgresql/<major>/docker`. Always mount the named volume to `/var/lib/postgresql` (not `/var/lib/postgresql/data`), otherwise container initialization fails.
- **Bind-Mount `is a directory` Error on Startup:** Ensure `infra/lgtm/otelcol-config.yaml` and `infra/lgtm/dashboards-provisioning.yaml` exist as regular files on the host before running `docker compose up`, or Docker will create empty directories at those paths and Grafana provisioning will exit.
- **Live Dashboard Edits:** `dashboards-provisioning.yaml` polls `/otel-lgtm/dashboards` every `10s`. Editing `infra/lgtm/dashboards/apprentice-stack.json` updates Grafana on browser refresh without restarting `apprentice-lgtm`.
