import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
CONTRACT_PATH = ROOT_DIR / "feature_contract.json"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

RE_NON_ALNUM = re.compile(r"[^a-z0-9\-_/\s]+")
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "to", "of", "in", "on", "for",
    "and", "or", "i", "my", "me", "you", "your", "we", "our", "can", "could",
    "would", "should", "do", "does", "did", "please", "help", "want", "need",
    "how", "what", "when", "where", "with", "that", "this", "it", "be", "have",
}


def load_contract():
    contract = json.loads(CONTRACT_PATH.read_text())
    canon = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    schema_hash = hashlib.sha256(canon.encode()).hexdigest()[:12]
    return contract, schema_hash


def tokenize(text: str) -> list[str]:
    clean = RE_NON_ALNUM.sub(" ", text.lower())
    raw = clean.split()
    tokens = [t for t in raw if t not in STOP_WORDS and len(t) > 1]
    return tokens if tokens else raw


def fnv1a_32(s: str) -> int:
    h = 2166136261
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def embed_text(text: str, dim: int = 256) -> np.ndarray:
    """Pure-math feature-hashed n-gram embedding (matches Go internal/index 1:1)."""
    vec = np.zeros(dim, dtype=np.float64)
    tokens = tokenize(text)
    for i, tok in enumerate(tokens):
        stem = tok
        for suf in ("ing", "ed", "s"):
            if len(stem) > len(suf) + 2 and stem.endswith(suf):
                stem = stem[: -len(suf)]
                break
        vec[fnv1a_32("w:" + stem) % dim] += 1.5
        if i + 1 < len(tokens):
            vec[fnv1a_32(f"b:{stem}_{tokens[i + 1]}") % dim] += 1.0
        for j in range(0, max(0, len(stem) - 2)):
            vec[fnv1a_32("c:" + stem[j : j + 3]) % dim] += 0.5
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


class BM25Index:
    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(d["indexed_text"]) for d in docs]
        self.doc_lens = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        self.tfs = [Counter(t) for t in self.doc_tokens]
        df = Counter()
        for tf in self.tfs:
            for term in tf:
                df[term] += 1
        n = len(docs)
        self.idf = {t: math.log(1.0 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, tenant_id: str, top_k: int) -> list[tuple[str, float]]:
        q_tokens = tokenize(query)
        scored = []
        for idx, doc in enumerate(self.docs):
            if doc["tenant_id"] != tenant_id:
                continue
            tf = self.tfs[idx]
            dl = self.doc_lens[idx]
            score = 0.0
            for qt in q_tokens:
                if qt not in tf:
                    continue
                f = tf[qt]
                idf = self.idf.get(qt, 0.0)
                score += idf * (f * (self.k1 + 1.0)) / (f + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl))
            if score > 0:
                scored.append((doc["chunk_id"], score))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_k]


class VectorIndex:
    def __init__(self, docs: list[dict], dim: int = 256):
        self.docs = docs
        self.dim = dim
        self.matrix = np.vstack([embed_text(d["indexed_text"], dim) for d in docs])

    def search(self, query: str, tenant_id: str, top_k: int) -> list[tuple[str, float]]:
        q_vec = embed_text(query, self.dim)
        sims = self.matrix @ q_vec
        scored = [
            (self.docs[i]["chunk_id"], float(sims[i]))
            for i in range(len(self.docs))
            if self.docs[i]["tenant_id"] == tenant_id
        ]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_k]


def fuse_rrf(bm25_hits: list[tuple[str, float]], vec_hits: list[tuple[str, float]], k: int = 60) -> list[tuple[str, float]]:
    scores = defaultdict(float)
    for rank, (cid, _) in enumerate(bm25_hits):
        scores[cid] += 1.0 / (k + rank + 1)
    for rank, (cid, _) in enumerate(vec_hits):
        scores[cid] += 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return fused


def load_bitext_dataframe() -> pd.DataFrame:
    ds_dir = Path(kagglehub.dataset_download("bitext/bitext-gen-ai-chatbot-customer-support-dataset"))
    csv_files = sorted(ds_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {ds_dir}")
    df = pd.read_csv(csv_files[0])
    df = df.dropna(subset=["category", "intent", "instruction", "response"]).reset_index(drop=True)
    return df


def build_policy_corpus_and_queries(df: pd.DataFrame):
    """
    Group real Bitext support rows by (category -> tenant_id, intent -> policy document).
    Pick 1 canonical response per intent as the authoritative policy chunk (with contextual header + exact policy_code),
    and hold out real customer instructions + exact policy-code lookups as the eval query set.
    """
    corpus = []
    eval_queries = []

    grouped = df.groupby(["category", "intent"], sort=True)
    tenant_counters = defaultdict(int)

    for (cat, intent), grp in grouped:
        tenant_id = str(cat).strip().upper()
        intent_slug = str(intent).strip().lower()
        tenant_counters[tenant_id] += 1
        policy_code = f"POL-{tenant_id[:4]}-{tenant_counters[tenant_id]:02d}"
        chunk_id = f"{policy_code}#0"

        # Pick the median-length real response as the clean canonical SOP text
        grp_sorted = grp.assign(rlen=grp["response"].str.len()).sort_values("rlen")
        canonical_resp = str(grp_sorted.iloc[len(grp_sorted) // 2]["response"]).strip()
        header = f"[Tenant: {tenant_id} | Policy: {policy_code} | Topic: {intent_slug.replace('_', ' ')}]"
        indexed_text = f"{header}\n{canonical_resp}"

        corpus.append({
            "chunk_id": chunk_id,
            "tenant_id": tenant_id,
            "policy_code": policy_code,
            "intent": intent_slug,
            "header": header,
            "content": canonical_resp,
            "indexed_text": indexed_text,
        })

        # Held-out Query Type A: 4 real customer paraphrases from Kaggle
        sample_instructions = grp["instruction"].drop_duplicates().head(4).tolist()
        for q in sample_instructions:
            eval_queries.append({
                "query": str(q).strip(),
                "tenant_id": tenant_id,
                "expected_chunk_id": chunk_id,
                "query_type": "paraphrase",
            })

        # Held-out Query Type B: 1 exact policy-code lookup (tests BM25 vs Vector exact-token blindness)
        eval_queries.append({
            "query": f"Explain the rules and procedure under {policy_code}",
            "tenant_id": tenant_id,
            "expected_chunk_id": chunk_id,
            "query_type": "exact_code",
        })

    return corpus, eval_queries


def evaluate(bm25: BM25Index, vec: VectorIndex, queries: list[dict], rrf_k: int, top_k: int = 5):
    modes = ["bm25_only", "vector_only", "hybrid_rrf"]
    stats = {m: {"hits": 0, "rr_sum": 0.0, "n": 0, "exact_hits": 0, "exact_n": 0, "para_hits": 0, "para_n": 0} for m in modes}

    for q in queries:
        b_hits = bm25.search(q["query"], q["tenant_id"], top_k=20)
        v_hits = vec.search(q["query"], q["tenant_id"], top_k=20)
        h_hits = fuse_rrf(b_hits, v_hits, k=rrf_k)

        ranked_lists = {
            "bm25_only": [cid for cid, _ in b_hits[:top_k]],
            "vector_only": [cid for cid, _ in v_hits[:top_k]],
            "hybrid_rrf": [cid for cid, _ in h_hits[:top_k]],
        }
        target = q["expected_chunk_id"]
        for mode, cids in ranked_lists.items():
            st = stats[mode]
            st["n"] += 1
            if q["query_type"] == "exact_code":
                st["exact_n"] += 1
            else:
                st["para_n"] += 1
            if target in cids:
                rank = cids.index(target) + 1
                st["hits"] += 1
                st["rr_sum"] += 1.0 / rank
                if q["query_type"] == "exact_code":
                    st["exact_hits"] += 1
                else:
                    st["para_hits"] += 1

    summary = []
    for mode in modes:
        st = stats[mode]
        summary.append({
            "strategy": mode,
            "recall_at_5": round(st["hits"] / max(1, st["n"]), 4),
            "mrr_at_5": round(st["rr_sum"] / max(1, st["n"]), 4),
            "exact_code_recall_at_5": round(st["exact_hits"] / max(1, st["exact_n"]), 4),
            "paraphrase_recall_at_5": round(st["para_hits"] / max(1, st["para_n"]), 4),
            "queries_evaluated": st["n"],
        })
    return summary


def main():
    contract, schema_hash = load_contract()
    ret = contract["retrieval"]

    print("Loading Kaggle dataset: bitext/bitext-gen-ai-chatbot-customer-support-dataset ...")
    df = load_bitext_dataframe()
    corpus, eval_queries = build_policy_corpus_and_queries(df)
    tenants = sorted({c["tenant_id"] for c in corpus})
    print(f"Built {len(corpus)} canonical policy chunks across {len(tenants)} tenants ({len(eval_queries)} eval queries)")

    bm25 = BM25Index(corpus, k1=ret["bm25_k1"], b=ret["bm25_b"])
    vec = VectorIndex(corpus, dim=ret["embed_dim"])
    benchmark_summary = evaluate(bm25, vec, eval_queries, rrf_k=ret["rrf_k"], top_k=ret["rerank_top_k"])

    print(f"\nProduction RAG Retrieval Audit | Schema: {contract['schema_version']} (hash={schema_hash})\n")
    for row in benchmark_summary:
        print(
            f"{row['strategy']:<14} | Recall@5: {row['recall_at_5']:.4f} | MRR@5: {row['mrr_at_5']:.4f} "
            f"| Exact-Code R@5: {row['exact_code_recall_at_5']:.4f} | Paraphrase R@5: {row['paraphrase_recall_at_5']:.4f}"
        )

    # Export corpus.jsonl for Go server + pgvector/RustFS ingestion
    with (ARTIFACTS_DIR / "corpus.jsonl").open("w") as f:
        for doc in corpus:
            f.write(json.dumps(doc) + "\n")

    (ARTIFACTS_DIR / "benchmark_summary.json").write_text(json.dumps(benchmark_summary, indent=2))

    # Export 12 deterministic golden retrieval fixtures for Python <-> Go parity tests
    golden_cases = []
    for q in eval_queries[:6] + [x for x in eval_queries if x["query_type"] == "exact_code"][:6]:
        b_hits = bm25.search(q["query"], q["tenant_id"], top_k=ret["bm25_top_k"])
        v_hits = vec.search(q["query"], q["tenant_id"], top_k=ret["vector_top_k"])
        h_hits = fuse_rrf(b_hits, v_hits, k=ret["rrf_k"])
        golden_cases.append({
            "query": q["query"],
            "tenant_id": q["tenant_id"],
            "query_type": q["query_type"],
            "expected_chunk_id": q["expected_chunk_id"],
            "top_rrf_chunk_id": h_hits[0][0] if h_hits else "",
            "top_rrf_score": round(h_hits[0][1], 6) if h_hits else 0.0,
        })

    golden = {
        "schema_version": contract["schema_version"],
        "schema_hash": schema_hash,
        "corpus_size": len(corpus),
        "tenants": tenants,
        "golden_cases": golden_cases,
    }
    (ARTIFACTS_DIR / "golden_fixtures.json").write_text(json.dumps(golden, indent=2))
    print(f"\nExported corpus.jsonl ({len(corpus)} chunks), benchmark_summary.json, and {len(golden_cases)} golden fixtures")


if __name__ == "__main__":
    main()
