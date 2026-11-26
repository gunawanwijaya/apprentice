# apprentice — Applied AI & Go Systems Engineering Lab

Production-oriented monorepo pairing **Python ML training, benchmarking & evaluation** (`uv`) with **deterministic, low-latency Go inference & control-plane services** (`go.mod` per module).

Every module enforces a **Zero Skew / Contract-First** engineering standard:
1. **`feature_contract.json`** — Single source of truth for feature order, types, nullability, preprocessing constants, pricing tables, and guardrail thresholds.
2. **Compile-Time Go Codegen (`go generate`)** — Emits typed Go structs and a canonical SHA-256 `SchemaHash` so Go services never rely on magic array indices or silent `null -> 0` coercions.
3. **Automated Cross-Language Parity & E2E Gates (`go test -count=1`)** — Verifies numerical parity (`<= 1e-4` / `1e-5`) and resilience invariants before any server binary ships.

---

## Module Index & Measured Benchmarks

| Module | Domain & Architecture | Real Datasets & Live Integrations | Go Runtime & Measured Latency | Verification Gate |
| :--- | :--- | :--- | :--- | :--- |
| **[`01-delivery-sla-breach`](./01-delivery-sla-breach)** | Tabular Classic ML (`LightGBM` -> `ONNX` -> Go) | Olist Brazilian E-Commerce (`95,992` orders) | Sub-ms CPU (`onnxruntime_go`), native `null <-> NaN` | **100/100 PASS** (`0.07s`, `<= 1e-5`) |
| **[`02-classic-nlp`](./02-classic-nlp)** | Classic NLP, BM25, NER & Deterministic PII Redaction | Bitext Support (`24.6k`), NER (`8k`), Multilingual Tickets | Pure-Go BM25 + Regex/Checksum PII + ONNX Intent | **Parity Verified** |
| **[`03-transformer-onnx`](./03-transformer-onnx)** | Neural Nets -> Transformers (`FP32` vs `INT8` Quantization Audit) | `bert-tiny` Fine-Tune + Held-Out Paraphrase Split | Pure-Go WordPiece + Go ONNX (`FP32: 1.15ms`, `INT8: 1.09ms`) | **100/100 PASS** (`0.49s`) |
| **[`04-multimodal-ai`](./04-multimodal-ai)** | Vision (`2D-CNN`), Audio VAD (`1D-CNN`) & Video (`ffmpeg` + `errgroup`) | SROIE Docs (`973`), Damaged Parcels (`8.5k`), Real Speech (`16.7k`) | Zero-CGO Pure-Go Pixels/PCM + Concurrent ONNX (`4.1–5.0ms`) | **60/60 PASS** (`0.07s`, `<= 1e-4`) |
| **[`05-llm-gateway`](./05-llm-gateway)** | Multi-Cloud LLM Control Plane: SSE, Entity-Safe Cache, Failover & Ledger | Live Groq (`gpt-oss-20b`) + Native Google Gemini + 3k-Token Audit | Pure-Go Zero-CGO (`64 µs` cache hit, `842ms` live JSON extract) | **4/4 E2E Suites + 10/10 Guard Parity** |
| **`06-production-rag`** *(In Progress)* | Hybrid RAG (`BM25` + `pgvector` HNSW + RRF + Rerank + Citations) | Multi-Tenant SOP & Technical Policy Corpus | Go Retrieval Orchestrator + Abstention & Citation Verifier | Recall@K, MRR & Grounding |
| **`07-agents-mcp-evals`** *(Planned)* | Bounded ReAct Tool Loops, MCP Server & Automated `LLM-as-a-Judge` Evals | Multi-Step Operations & Incident Triage Scenarios | Go MCP Server + Deterministic CI Eval Harness | CI Regression Gate |

---

## Monorepo Tooling Rules

* **Isolated Python Environments:** Each `<module>/ml` owns its own `.python-version`, `pyproject.toml`, and `uv.lock` (preventing C-extension or CUDA/CPU dependency clashes across `lightgbm`, `torch`, and `onnxruntime`).
* **Isolated Go Modules:** Each `<module>/go` compiles cleanly with `GOWORK=off` to guarantee zero cross-module coupling in CI.

```bash
# Run any module from the repository root:
make train MOD=05-llm-gateway
make gen   MOD=05-llm-gateway
make test  MOD=05-llm-gateway
make serve MOD=05-llm-gateway
```