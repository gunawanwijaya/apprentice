# 02 — Classic NLP, PII Redaction, BM25 & Statistical NER (Python → ONNX + Pure-Go Viterbi)

A sub-millisecond, zero-LLM **Classic NLP & NLU Engine** written in Go and trained in Python. It combines **ONNX Runtime** (for end-to-end TF-IDF + linear text classification) with **deterministic pure-Go runtimes** (for Indonesian PII redaction, BM25Okapi keyword retrieval, and CRF Viterbi sequence decoding)—all gated by a single SHA-256 `feature_contract.json` and a 5-way Python ↔ Go parity test suite (`0.132s`).

---

## Why Classic NLP Before Reaching for an LLM

In production e-commerce and fintech backends, routing 100% of support chat, ticket triage, and PII scrubbing through hosted LLM APIs creates three avoidable failure modes:

1. **Privacy Boundary Violation (Fintech Compliance):** You cannot send raw customer chat containing Indonesian NIK/KTP numbers or `08`/`628` phone numbers to an external LLM API just to ask it to redact PII—the data has already left your network boundary. Drawing from financial-grade PII protection standards (as enforced in fintech lending backends like KoinWorks), structured PII scrubbing must execute locally at ingress in microseconds.
2. **Latency & Unit Economics at Scale:** Classifying a support message (`get_refund`, `track_order`) or routing a multilingual helpdesk ticket via CPU-served TF-IDF + Logistic Regression takes **290–401 µs** on a single CPU core for effectively $0 per million requests, compared to 500–2,000 ms and recurring token costs for an LLM.
3. **Exact-Match Blind Spots in Pure Vector Search:** Dense embeddings capture semantic similarity but quietly degrade on exact identifiers (order IDs, SKUs, error codes, policy terms). **BM25Okapi** nails exact lexical matches in **92 µs** and forms the lexical half of the Hybrid RAG (BM25 + Dense Vectors + RRF) architecture in Module 06.

---

## Hybrid Serving Architecture (`train.py` ↔ `main.go`)

This module showcases **two** complementary production Go ML serving patterns inside a single HTTP service, governed by one rule: **Python owns all training, evaluation, and artifact emission; Go only executes deterministic runtime mechanics.**

```text
feature_contract.json (Single Source of Truth: token_pattern, BM25 k1/b/top_k, model manifest)
 ├── ml/train.py
 │    ├── 1. skl2onnx TF-IDF Skew Audit (proves & fixes RE2 vs Python re divergence)
 │    ├── 2. Bitext Intent Clf + FAQ Corpus -> artifacts/intent.onnx + artifacts/bm25_corpus.json
 │    ├── 3. BIO NER CRF (sklearn-crfsuite) -> artifacts/ner_crf.json (state + transition tables)
 │    ├── 4. Multilingual Ticket Triage     -> artifacts/queue.onnx + artifacts/priority.onnx
 │    └── 5. Unified Parity Manifest        -> artifacts/golden_fixtures.json (schema_hash=ad62d4533bea)
 │
 └── go generate (internal/contract/gen.go) -> internal/contract/contract_gen.go
      └── go test -count=1 (parity_test.go) -> 5/5 subtests PASS in 0.132s
           ├── intent_onnx      (50 fixtures, |go_prob - py_prob| <= 1e-4)
           ├── bm25_pure_go     (15 fixtures, exact top-K doc ID match vs rank_bm25)
           ├── ner_crf_viterbi  (50 sentences, 100% exact BIO tag sequence match vs sklearn-crfsuite)
           ├── triage_onnx      (30 multilingual fixtures, queue + priority parity)
           └── pii_redactor     (deterministic Indonesian phone, 16-digit NIK/KTP, email masking)
```

### Pattern A: Bundled ONNX Graphs (`intent.onnx`, `queue.onnx`, `priority.onnx`)
Scikit-learn `Pipeline([TfidfVectorizer, LogisticRegression])` models are compiled into self-contained `.onnx` graphs accepting raw UTF-8 `StringTensor` inputs (`[1, 1]`). Go passes redacted text directly into `yalue/onnxruntime_go` without re-implementing TF-IDF tokenization or vocabulary hashing in Go.

### Pattern B: Zero-CGO Pure-Go Deterministic Runtime (`viterbi`, `bm25`, `pii`)
Standard ONNX has no native Conditional Random Field (CRF) / Viterbi operator. Rather than maintaining a brittle C++ custom ONNX operator or wrapping Python in a sidecar container:
* Python trains the L-BFGS CRF (`sklearn-crfsuite`) and serializes only the learned lookup tables (`11,799` state feature weights and `217` label transition weights) to `artifacts/ner_crf.json`.
* Go (`internal/viterbi`) mirrors the deterministic token-shape feature extractor (`lower`, `suf2`, `suf3`, `isupper`, `istitle`, `isdigit`, `BOS`/`EOS`, ±1 window) and runs a ~60-line **Viterbi Dynamic Programming** decoder to find the maximum-scoring BIO tag path—the same family of deterministic shortest-path/DP graph mechanics used in multi-hop 3PL logistics routing (e.g., A* pathfinding across fulfillment legs).

---

## Deep-Dive: The `skl2onnx` TF-IDF Tokenizer Skew Audit

A common silent failure when exporting `scikit-learn` NLP pipelines to ONNX is **regex engine drift** between Python's `re` module and ONNX Runtime's Google **RE2** `Tokenizer` operator:

| Divergence Cause | `scikit-learn` Default (`(?u)\b\w\w+\b`) | `skl2onnx` Default Rewrite (`[a-zA-Z0-9_]+`) | Production Impact |
| :--- | :--- | :--- | :--- |
| **1-Character Tokens** (`"I"`, `"a"`, `"5"`) | Dropped (`\w\w+` requires $\ge 2$ chars) | Kept (`+` matches $\ge 1$ char) | Shifts TF-IDF L2 norm and predicted probabilities |
| **Non-ASCII / Unicode** (`"für"`, `"contraseña"`) | Matched intact via `(?u)` Unicode `\w` | Split at accented characters by ASCII class | Breaks German/Spanish/French words on multilingual tickets |

### Proof & Fix Built Directly Into `ml/train.py`
Before training any production model, `train.py` runs `run_tfidf_skew_audit()` on edge-case inputs (`"I want a refund for order 5"`, `"Ich brauche Hilfe für mein Konto"`, `"No puedo cambiar mi contraseña"`):

```text
SKL2ONNX TF-IDF TOKENIZER SKEW AUDIT
==========================================================================
1) Naive default TfidfVectorizer max |sklearn - onnx| diff : 0.010035 (SKEW!)
2) Locked token_pattern='[a-zA-Z0-9_]+' max diff        : 0.000000 (PARITY OK)
```

**Resolution:** `feature_contract.json` locks `"token_pattern": "[a-zA-Z0-9_]+"` across both `TfidfVectorizer(token_pattern=...)` and `skl2onnx`'s `options={TfidfVectorizer: {"tokenexp": ...}}` (as well as Go's `bm25` tokenizer), and `train.py` asserts `max_diff < 1e-4` between `sklearn` and `onnxruntime` before writing `golden_fixtures.json`.

---

## Datasets, Benchmarks & Parity Results

All three real-world Kaggle datasets download automatically via `kagglehub` (with built-in synthetic fallbacks for offline CI).

### 1. Support Intent Classification + BM25 FAQ Search
* **Dataset:** `bitext/bitext-gen-ai-chatbot-customer-support-dataset` (`26,872` rows → **`24,635`** unique instructions after deduplicating ~8.3% exact duplicates to prevent train/test leakage; **27 intents** across order, refund, payment, and delivery flows).
* **Export Choice:** `LogisticRegression` exported to `intent.onnx` for calibrated probabilities alongside clean `zipmap=False` ONNX tensor output (`max_py_vs_onnx_diff = 2.64e-07`).

| Model | Accuracy | Macro-F1 | Exported? |
| :--- | :---: | :---: | :---: |
| `LinearSVC` | `0.9939` | `0.9936` | Benchmark only (uncalibrated margins) |
| **`LogisticRegression`** | **`0.9919`** | **`0.9915`** | **Yes (`artifacts/intent.onnx`)** |
| `MultinomialNB` | `0.9854` | `0.9840` | Baseline |

* **BM25 Corpus:** 27 representative FAQ documents (`artifacts/bm25_corpus.json`, `k1=1.5`, `b=0.75`), 15/15 golden queries matched 1:1 against Python `rank_bm25.BM25Okapi`.

### 2. Token-Level Named Entity Recognition (BIO Tagging)
* **Dataset:** `naseralqaydeh/named-entity-recognition-ner-corpus` (`8,000` sentences, **17 BIO tags** including `B-per`, `I-per`, `B-org`, `B-geo`, `B-tim`, `O`).
* **Model:** `sklearn-crfsuite` L-BFGS (`c1=0.1`, `c2=0.1`, `max_iterations=80`).
* **Entity Micro-F1 (excluding `O`):** **`0.8114`** (`11,799` state features, `217` transitions).
* **Go Parity:** **50/50** held-out test sentences matched token-for-token in pure-Go Viterbi (`0.03s`).
* *Domain-Shift Note:* On the Indonesian smoke test (`"Agung works at Tokopedia in Jakarta"`), the model correctly tags `Jakarta` (`B-geo`) and detects `Tokopedia` as an entity (`B-geo`), while tagging `Agung` as `O` because the news training corpus has negligible Indonesian personal-name coverage—illustrating why production Indonesian PII pipelines layer deterministic regex + domain-specific NER or local zero-shot NER (e.g., GLiNER) rather than relying on English news corpora alone.

### 3. Multilingual Ticket Triage (`EN` / `DE` / `ES` / `FR`)
* **Dataset:** `tobiasbueck/email-ticket-text-german-classification` (`200` rows, 3 queues, 3 priority levels).
* **Queue Classifier (`queue.onnx`):** Macro-F1 **`1.0000`** (`max_py_vs_onnx_diff = 9.20e-08`).
* **Priority Classifier (`priority.onnx`):** Macro-F1 **`0.6283`** (`max_py_vs_onnx_diff = 5.07e-08`) — reflects expected ambiguity in short urgency labels with limited training rows.

---

## Live Go Server Endpoints & Measured Latency

All inference endpoints run strict JSON ingress (`DisallowUnknownFields`, 64 KB body cap), `log/slog` structured JSON logs, graceful `SIGINT`/`SIGTERM` shutdown, and automatic ingress PII redaction before text hits ONNX models.

| Endpoint | Engine | Observed Latency | What It Does |
| :--- | :--- | :---: | :--- |
| `GET /healthz` | Pure Go | `< 50 µs` | Returns `status`, `schema_version`, and `schema_hash` (`ad62d4533bea`) |
| `POST /redact` | Pure Go (`internal/pii`) | **`91 µs`** | Masks emails (`[EMAIL]`), 16-digit NIK/KTP (`[KTP]`), and `08`/`+628` phones (`[PHONE]`) |
| `POST /search` | Pure Go (`internal/bm25`) | **`92 µs`** | Ranks top-K FAQ responses via BM25Okapi over `bm25_corpus.json` |
| `POST /ner` | Pure Go (`internal/viterbi`) | **`123 µs`** | Decodes optimal BIO entity tag sequence over `ner_crf.json` |
| `POST /triage` | ONNX Runtime (`queue` + `priority`) | **`290 µs`** | Redacts PII, then runs dual ONNX inference for department queue & priority |
| `POST /intent` | ONNX Runtime (`intent.onnx`) | **`401 µs`** | Redacts PII, then classifies support intent across 27 classes |

---

## Repository Layout

```text
02-classic-nlp/
├── README.md
├── feature_contract.json          # Locked token_pattern, BM25 params, model manifest
├── artifacts/
│   ├── intent.onnx                # Bitext 27-class TF-IDF + LogisticRegression
│   ├── queue.onnx                 # Multilingual ticket queue classifier
│   ├── priority.onnx              # Multilingual ticket priority classifier
│   ├── ner_crf.json               # Learned CRF state & transition weights
│   ├── bm25_corpus.json           # Tokenized FAQ corpus for pure-Go BM25
│   └── golden_fixtures.json       # Unified 4-task parity fixtures + schema_hash
├── ml/
│   ├── .python-version            # 3.12
│   ├── pyproject.toml
│   ├── uv.lock
│   └── train.py                   # Skew audit + 3-dataset training + ONNX/JSON export
└── go/
    ├── go.mod
    ├── go.sum
    ├── parity_test.go             # 5-way Python <-> Go parity gate
    ├── cmd/server/main.go         # Unified HTTP server (6 flat endpoints)
    └── internal/
        ├── contract/              # go:generate -> contract_gen.go (SHA-256 hash)
        ├── pii/                   # Pure-Go Indonesian PII redactor
        ├── bm25/                  # Pure-Go BM25Okapi scorer
        └── viterbi/               # Pure-Go CRF Viterbi DP decoder
```

---

## Quickstart

From the `apprentice/` repository root:

```bash
# 1. Run skew audit, train all 3 datasets, and export artifacts + golden_fixtures.json
make train MOD=02-classic-nlp

# 2. Regenerate Go contract constants & SHA-256 schema_hash
make gen MOD=02-classic-nlp

# 3. Execute non-cached 5-way Python <-> Go parity test suite
make test MOD=02-classic-nlp

# 4. Start the Go HTTP inference server on :8082
make serve MOD=02-classic-nlp
```

### Example Requests

```bash
# Ingress PII redaction (91 us)
curl -s -X POST http://localhost:8082/redact \
  -H "Content-Type: application/json" \
  -d '{"text":"Hubungi 081234567890, NIK 3171012345678901, email budi@example.com"}' | jq

# PII-redacted intent classification (401 us)
curl -s -X POST http://localhost:8082/intent \
  -H "Content-Type: application/json" \
  -d '{"text":"I want a refund for my order, call me at 081234567890"}' | jq

# Pure-Go BM25 FAQ retrieval (92 us)
curl -s -X POST http://localhost:8082/search \
  -H "Content-Type: application/json" \
  -d '{"query":"cancel order refund"}' | jq

# Pure-Go CRF Viterbi NER (123 us)
curl -s -X POST http://localhost:8082/ner \
  -H "Content-Type: application/json" \
  -d '{"text":"Agung works at Tokopedia in Jakarta"}' | jq

# Multilingual ticket triage (290 us)
curl -s -X POST http://localhost:8082/triage \
  -H "Content-Type: application/json" \
  -d '{"text":"Ich brauche Hilfe für mein Konto, dringend!"}' | jq
```
