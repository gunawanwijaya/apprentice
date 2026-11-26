# apprentice

Applied AI & Go engineering lab. Every module pairs a reproducible **Python (`uv`) training & evaluation pipeline** with a **zero-skew Go inference runtime**, governed by a single `feature_contract.json`, SHA-256 schema hashing, and cross-language `golden_fixtures.json` parity gates.

## Modules

| # | Module | Domain & Stack | Serving Pattern | Parity Gate | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **01** | [`01-delivery-sla-breach`](./01-delivery-sla-breach) | E-Commerce SLA Breach (Olist, LightGBM / XGBoost / CatBoost) | ONNX Runtime (`model.onnx`) + `go:generate` typed contract | `100/100` rows `< 1e-5` (`0.07s`) | **Shipped** |
| **02** | [`02-classic-nlp`](./02-classic-nlp) | Support Intent, Multilingual Triage, BIO NER & FAQ Search | Hybrid: ONNX Runtime (`intent`, `queue`, `priority`) + Pure-Go (`pii`, `bm25`, `viterbi`) | `5/5` pipelines PASS (`0.13s`) | **Shipped** |
| **03** | `03-transformer-onnx` | Lightweight Transformer Encoder (MiniLM / DistilBERT) | Quantized INT8 ONNX + Go WordPiece tokenizer | Planned | Up Next |
| **04** | `04-multimodal-pipeline` | Receipt Vision/OCR & Audio STT | Tiered local OCR + Vision LLM fallback in Go | Planned | Planned |
| **05** | `05-go-llm-gateway` | Production LLM Proxy (routing, structured JSON, cost/token logs) | Pure Go HTTP Gateway + PostgreSQL | Planned | Planned |
| **06** | `06-hybrid-rag-pgvector` | Hybrid RAG (`pgvector` + BM25 + RRF + grounded citations) | Go + PostgreSQL `pgvector` | Planned | Planned |
| **07** | `07-mcp-agent-evals` | Tool-Calling Agent, MCP Server & Automated Eval Suite | Go MCP Server + LLM-as-a-Judge eval harness | Planned | Planned |

## Engineering Standards Across All Modules

1. **Strict Environment Isolation:** Each module owns its own `ml/pyproject.toml` (`uv.lock`) and `go/go.mod`—no shared root `uv` workspace, and CI verifies Go modules with `GOWORK=off`.
2. **Contract-First Parity:** Python owns all training, benchmarking, and artifact emission; Go executes deterministic runtime mechanics and refuses to pass `make test` if `schema_hash` or golden outputs drift.
3. **Uniform CLI Workflow:**
   ```bash
   make train MOD=<module>   # Train, benchmark, and export artifacts + golden_fixtures.json
   make gen   MOD=<module>   # Regenerate Go contract code & SHA-256 schema_hash
   make test  MOD=<module>   # Run non-cached Python <-> Go parity test suite
   make serve MOD=<module>   # Start the Go HTTP inference server on :8080
   ```
