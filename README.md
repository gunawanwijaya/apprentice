# apprentice — Applied AI & Go Systems Engineering Lab

Production-oriented monorepo pairing **Python ML training & evaluation** (`uv`) with **deterministic, sub-millisecond Go inference & API services** (`go.mod` per module).

Every ML module enforces a **Zero Training-Serving Skew** contract:
1. **`feature_contract.json`** — Single source of truth for feature order, types, nullability, preprocessing constants, and decision thresholds.
2. **Compile-Time Go Codegen (`go generate`)** — Emits typed structs and a canonical SHA-256 `SchemaHash` so Go services never rely on magic array indices or silent `null -> 0` coercions.
3. **Cross-Language Golden Parity Gate (`go test -count=1`)** — Asserts 1:1 numerical parity between Python (`onnxruntime` / `torch` / `lightgbm`) and Go (`onnxruntime_go` + pure-Go preprocessors) on real-world holdout fixtures before a server binary is allowed to ship.

---

## Module Index & Measured Benchmarks

| Module | Domain & Architecture | Real-World Datasets | Go Runtime & Latency | Parity Gate |
| :--- | :--- | :--- | :--- | :--- |
| **[`01-delivery-sla-breach`](./01-delivery-sla-breach)** | Tabular Classic ML (`LightGBM` -> `ONNX` -> Go) | Olist Brazilian E-Commerce (`95,992` orders) | Sub-ms CPU (`onnxruntime_go`), native `null <-> NaN` | **100/100 PASS** (`0.07s`, `<= 1e-5`) |
| **[`02-classic-nlp`](./02-classic-nlp)** | Classic NLP, BM25, NER & Deterministic PII Redaction | Bitext Support (`24.6k`), NER (`8k`), Multilingual Tickets | Pure-Go BM25 + Regex/Checksum PII + ONNX Intent | **Parity Verified** |
| **[`03-transformer-onnx`](./03-transformer-onnx)** | Neural Nets -> Transformers (`FP32` vs `INT8` Quantization Audit) | `bert-tiny` Fine-Tune + Held-Out Paraphrase Split | Pure-Go WordPiece + Go ONNX (`FP32: 1.15ms`, `INT8: 1.09ms`) | **100/100 PASS** (`0.49s`) |
| **[`04-multimodal-ai`](./04-multimodal-ai)** | Vision (`2D-CNN`), Audio VAD (`1D-CNN`) & Video (`ffmpeg` + `errgroup`) | SROIE Docs (`973`), Damaged Parcels (`8.5k`), Real Speech (`16.7k`) | Zero-CGO Pure-Go Pixels/PCM + Concurrent ONNX (`4.1–5.0ms`) | **60/60 PASS** (`0.07s`, `<= 1e-4`) |
| **`05-llm-gateway`** *(In Progress)* | LLM API Gateway: Streaming SSE, Structured JSON, Rate Limits, Failover & Cost Ledger | Multi-Provider OpenAI-Compatible + Local Mock/Live Upstream | Pure-Go `net/http` SSE Proxy, Context Cancellation, Exact/Semantic Cache | Contract & Replay Suite |
| **`06-production-rag`** *(Planned)* | Hybrid RAG (`BM25` + Dense Vectors + Cross-Encoder Rerank + Citations) | Technical Documentation Corpus | Go Retrieval Orchestrator + PostgreSQL `pgvector` | Retrieval Recall@K & Grounding |
| **`07-agents-mcp-evals`** *(Planned)* | Bounded ReAct Tool Loops, MCP Server & Automated `LLM-as-a-Judge` Evals | Multi-Step Operations & Triage Scenarios | Go MCP Server + Deterministic Eval Harness | CI Regression Gate |

---

## Monorepo Tooling Rules

* **Isolated Python Environments:** Each `<module>/ml` owns its own `.python-version`, `pyproject.toml`, and `uv.lock` (no shared root virtualenv conflicts between `lightgbm`, `torch`, and `onnxruntime`).
* **Isolated Go Modules:** Each `<module>/go` compiles cleanly with `GOWORK=off` to guarantee zero cross-module coupling in CI.

```bash
# Run any module from the repository root:
make train MOD=04-multimodal-ai
make gen   MOD=04-multimodal-ai
make test  MOD=04-multimodal-ai
make serve MOD=04-multimodal-ai
```
