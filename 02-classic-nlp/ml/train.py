import ast
import hashlib
import json
import re
from pathlib import Path

import kagglehub
import numpy as np
import onnxruntime as ort
import pandas as pd
import sklearn_crfsuite
from rank_bm25 import BM25Okapi
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT_DIR = Path(__file__).resolve().parent.parent
CONTRACT_PATH = ROOT_DIR / "feature_contract.json"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"


def load_contract() -> tuple[dict, str]:
    contract = json.loads(CONTRACT_PATH.read_text())
    canonical = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    schema_hash = hashlib.sha256(canonical).hexdigest()[:12]
    return contract, schema_hash


def download_csv(slug: str) -> Path | None:
    try:
        folder = Path(kagglehub.dataset_download(slug))
        csvs = sorted(folder.rglob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
        return csvs[0] if csvs else None
    except Exception as e:
        print(f"[warn] kagglehub skipped for {slug} ({e}); using synthetic fallback")
        return None


def run_tfidf_skew_audit(token_pattern: str) -> None:
    """Demonstrate why default sklearn TfidfVectorizer diverges from skl2onnx RE2, and prove the fix."""
    print("=" * 74)
    print("SKL2ONNX TF-IDF TOKENIZER SKEW AUDIT")
    print("=" * 74)
    samples = [
        "I want a refund for order 5",          # 1-char tokens: 'I', 'a', '5'
        "Ich brauche Hilfe für mein Konto",     # Non-ASCII: 'für'
        "No puedo cambiar mi contraseña",       # Non-ASCII: 'contraseña'
        "track my order status right now",      # Clean ASCII control
    ]
    labels = [0, 1, 1, 0]

    # 1. Naive default sklearn (token_pattern=(?u)\b\w\w+\b) -> RE2 rewrites to [a-zA-Z0-9_]+
    naive_pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=500)),
    ]).fit(samples, labels)
    naive_onnx = convert_sklearn(
        naive_pipe,
        initial_types=[("input", StringTensorType([None, 1]))],
        options={LogisticRegression: {"zipmap": False}},
    )
    sess_naive = ort.InferenceSession(naive_onnx.SerializeToString(), providers=["CPUExecutionProvider"])
    py_p = naive_pipe.predict_proba(samples)[:, 1]
    ox_p = sess_naive.run(None, {"input": np.array([[s] for s in samples])})[1][:, 1]
    max_naive_diff = float(np.max(np.abs(py_p - ox_p)))
    print(f"1) Naive default TfidfVectorizer max |sklearn - onnx| diff : {max_naive_diff:.6f} (SKEW!)")

    # 2. Fixed: lock identical RE2-compatible token_pattern on both sides
    fixed_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(token_pattern=token_pattern, lowercase=True)),
        ("clf", LogisticRegression(max_iter=500)),
    ]).fit(samples, labels)
    fixed_onnx = convert_sklearn(
        fixed_pipe,
        initial_types=[("input", StringTensorType([None, 1]))],
        options={
            TfidfVectorizer: {"tokenexp": token_pattern},
            LogisticRegression: {"zipmap": False},
        },
    )
    sess_fixed = ort.InferenceSession(fixed_onnx.SerializeToString(), providers=["CPUExecutionProvider"])
    py_f = fixed_pipe.predict_proba(samples)[:, 1]
    ox_f = sess_fixed.run(None, {"input": np.array([[s] for s in samples])})[1][:, 1]
    max_fixed_diff = float(np.max(np.abs(py_f - ox_f)))
    print(f"2) Locked token_pattern='{token_pattern}' max diff        : {max_fixed_diff:.6f} (PARITY OK)\n")
    assert max_fixed_diff < 1e-5, f"Expected parity after locking token_pattern, got {max_fixed_diff}"


def export_text_clf_onnx(pipe: Pipeline, token_pattern: str, out_path: Path, sample_texts: list[str]) -> list[dict]:
    onnx_model = convert_sklearn(
        pipe,
        initial_types=[("input", StringTensorType([None, 1]))],
        options={
            TfidfVectorizer: {"tokenexp": token_pattern},
            LogisticRegression: {"zipmap": False},
        },
    )
    out_path.write_bytes(onnx_model.SerializeToString())

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    classes = [str(c) for c in pipe.classes_]
    fixtures = []
    max_diff = 0.0

    for text in sample_texts:
        py_probs = pipe.predict_proba([text])[0]
        ox_labels, ox_probs = sess.run(None, {"input": np.array([[text]])})
        ox_prob_vec = ox_probs[0]
        diff = float(np.max(np.abs(py_probs - ox_prob_vec)))
        max_diff = max(max_diff, diff)
        top_idx = int(np.argmax(ox_prob_vec))
        fixtures.append({
            "text": text,
            "expected_label": classes[top_idx],
            "expected_prob": round(float(ox_prob_vec[top_idx]), 6),
        })

    assert max_diff < 1e-4, f"{out_path.name} parity check failed: max_diff={max_diff}"
    print(f"Exported {out_path.name} (classes={len(classes)}, max_py_vs_onnx_diff={max_diff:.2e})")
    return fixtures


def train_intent_and_bm25(contract: dict) -> tuple[list[dict], list[dict]]:
    print("=" * 74)
    print("TASK 1: SUPPORT INTENT CLASSIFICATION + BM25 FAQ SEARCH (Bitext)")
    print("=" * 74)
    token_pattern = contract["token_pattern"]
    csv_path = download_csv("bitext/bitext-gen-ai-chatbot-customer-support-dataset")

    if csv_path is not None:
        df = pd.read_csv(csv_path)[["instruction", "intent", "response"]].dropna()
        before = len(df)
        df = df.drop_duplicates(subset=["instruction"]).reset_index(drop=True)
        print(f"Loaded Bitext ({before:,} rows -> {len(df):,} after deduplication | {df['intent'].nunique()} intents)")
    else:
        df = pd.DataFrame({
            "instruction": [
                "where is my order", "track my package", "check delivery status",
                "i want a refund", "return my item", "cancel order and refund",
                "payment failed", "credit card declined", "update billing method",
            ] * 40,
            "intent": (["track_order"] * 3 + ["get_refund"] * 3 + ["payment_issue"] * 3) * 40,
            "response": (["Track your order in My Orders."] * 3 + ["Submit a refund request."] * 3 + ["Check your card details."] * 3) * 40,
        })

    X_train, X_test, y_train, y_test = train_test_split(
        df["instruction"].astype(str).tolist(),
        df["intent"].astype(str).tolist(),
        test_size=0.2,
        stratify=df["intent"].astype(str).tolist(),
        random_state=42,
    )

    candidates = {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000, C=2.0),
        "LinearSVC": LinearSVC(random_state=42),
    }
    results = []
    trained = {}
    for name, clf in candidates.items():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(token_pattern=token_pattern, lowercase=True, ngram_range=(1, 2), max_features=15000)),
            ("clf", clf),
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        trained[name] = pipe
        results.append((name, accuracy_score(y_test, preds), f1_score(y_test, preds, average="macro")))

    results.sort(key=lambda r: r[2], reverse=True)
    print(f"{'Model':20s} | {'Accuracy':>8s} | {'Macro-F1':>8s}")
    print("-" * 44)
    for name, acc, mf1 in results:
        print(f"{name:20s} | {acc:8.4f} | {mf1:8.4f}")

    # Export LogisticRegression (calibrated probabilities + clean ONNX support)
    intent_fixtures = export_text_clf_onnx(
        trained["LogisticRegression"], token_pattern, ARTIFACTS_DIR / "intent.onnx", X_test[:50]
    )

    # Build compact BM25 FAQ corpus (1 representative doc per intent) + golden search queries
    faq_df = df.groupby("intent", as_index=False).first()[["intent", "instruction", "response"]]
    tok_re = re.compile(token_pattern)
    docs = []
    tokenized_corpus = []
    for _, row in faq_df.iterrows():
        doc_text = f"{row['intent'].replace('_', ' ')} {row['instruction']} {row['response']}"
        tokens = tok_re.findall(doc_text.lower())
        tokenized_corpus.append(tokens)
        docs.append({
            "id": str(row["intent"]),
            "text": doc_text,
            "response": str(row["response"]),
            "tokens": tokens,
        })

    (ARTIFACTS_DIR / "bm25_corpus.json").write_text(json.dumps({
        "k1": contract["bm25"]["k1"],
        "b": contract["bm25"]["b"],
        "docs": docs,
    }, indent=2))

    bm25 = BM25Okapi(tokenized_corpus, k1=contract["bm25"]["k1"], b=contract["bm25"]["b"])
    bm25_fixtures = []
    for q in X_test[:15]:
        q_tokens = tok_re.findall(q.lower())
        scores = bm25.get_scores(q_tokens)
        top_idxs = np.argsort(-scores)[: contract["bm25"]["top_k"]]
        bm25_fixtures.append({
            "query": q,
            "expected_top_ids": [docs[int(i)]["id"] for i in top_idxs if scores[int(i)] > 0],
        })
    print(f"Exported bm25_corpus.json ({len(docs)} FAQ docs) and {len(bm25_fixtures)} BM25 fixtures\n")
    return intent_fixtures, bm25_fixtures


def word_features(tokens: list[str], i: int) -> dict[str, float]:
    """Deterministic feature extractor mirrored 1:1 in Go internal/viterbi."""
    w = tokens[i]
    feats = {
        "bias": 1.0,
        f"w.lower={w.lower()}": 1.0,
        f"w.suf3={w[-3:]}": 1.0,
        f"w.suf2={w[-2:]}": 1.0,
        f"w.isupper={int(w.isupper())}": 1.0,
        f"w.istitle={int(w.istitle())}": 1.0,
        f"w.isdigit={int(w.isdigit())}": 1.0,
    }
    if i > 0:
        pw = tokens[i - 1]
        feats[f"-1:w.lower={pw.lower()}"] = 1.0
        feats[f"-1:w.istitle={int(pw.istitle())}"] = 1.0
    else:
        feats["BOS"] = 1.0
    if i < len(tokens) - 1:
        nw = tokens[i + 1]
        feats[f"+1:w.lower={nw.lower()}"] = 1.0
        feats[f"+1:w.istitle={int(nw.istitle())}"] = 1.0
    else:
        feats["EOS"] = 1.0
    return feats


def train_ner_crf() -> list[dict]:
    print("=" * 74)
    print("TASK 2: TOKEN-LEVEL NER (sklearn-crfsuite -> JSON -> Pure-Go Viterbi)")
    print("=" * 74)
    csv_path = download_csv("naseralqaydeh/named-entity-recognition-ner-corpus")
    sentences: list[tuple[list[str], list[str]]] = []

    if csv_path is not None:
        df = pd.read_csv(csv_path, encoding="latin1")
        if "Sentence" in df.columns and "Tag" in df.columns:
            for _, row in df.dropna(subset=["Sentence", "Tag"]).head(8000).iterrows():
                toks = str(row["Sentence"]).split()
                tags = ast.literal_eval(str(row["Tag"])) if str(row["Tag"]).startswith("[") else str(row["Tag"]).split()
                if len(toks) == len(tags) and len(toks) > 0:
                    sentences.append((toks, tags))
        elif "Word" in df.columns and "Tag" in df.columns:
            df["Sentence #"] = df["Sentence #"].ffill()
            for _, grp in list(df.groupby("Sentence #"))[:8000]:
                sentences.append((grp["Word"].astype(str).tolist(), grp["Tag"].astype(str).tolist()))
        print(f"Loaded NER corpus ({len(sentences):,} sentences)")
    else:
        sentences = [
            (["Agung", "works", "at", "Tokopedia", "in", "Jakarta"], ["B-per", "O", "O", "B-org", "O", "B-geo"]),
            (["Budi", "shipped", "to", "Tokyo", "via", "JNE"], ["B-per", "O", "O", "B-geo", "O", "B-org"]),
        ] * 100

    train_s, test_s = train_test_split(sentences, test_size=0.2, random_state=42)
    X_train = [[word_features(toks, i) for i in range(len(toks))] for toks, _ in train_s]
    y_train = [tags for _, tags in train_s]
    X_test = [[word_features(toks, i) for i in range(len(toks))] for toks, _ in test_s]
    y_test = [tags for _, tags in test_s]

    crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=80, all_possible_transitions=True)
    crf.fit(X_train, y_train)
    y_pred = crf.predict(X_test)

    flat_true = [t for seq in y_test for t in seq]
    flat_pred = [t for seq in y_pred for t in seq]
    non_o_labels = sorted({t for t in flat_true if t != "O"})
    entity_f1 = f1_score(flat_true, flat_pred, labels=non_o_labels, average="micro", zero_division=0)
    print(f"CRF Entity Micro-F1 (excluding 'O'): {entity_f1:.4f} | Labels: {len(crf.classes_)}")

    # Serialize state + transition weights for deterministic pure-Go Viterbi decoding
    state_weights: dict[str, dict[str, float]] = {}
    for (attr, label), weight in crf.state_features_.items():
        if abs(weight) > 1e-4:
            state_weights.setdefault(attr, {})[label] = round(float(weight), 6)

    transition_weights: dict[str, dict[str, float]] = {}
    for (l_from, l_to), weight in crf.transition_features_.items():
        transition_weights.setdefault(l_from, {})[l_to] = round(float(weight), 6)

    ner_artifact = {
        "labels": list(crf.classes_),
        "state_weights": state_weights,
        "transition_weights": transition_weights,
    }
    (ARTIFACTS_DIR / "ner_crf.json").write_text(json.dumps(ner_artifact))
    print(f"Exported ner_crf.json (state_features={len(state_weights):,}, transitions={len(crf.transition_features_)})\n")

    return [
        {"tokens": toks, "expected_tags": pred_tags}
        for (toks, _), pred_tags in zip(test_s[:50], y_pred[:50])
    ]


def train_multilingual_triage(contract: dict) -> list[dict]:
    print("=" * 74)
    print("TASK 3: MULTILINGUAL TICKET TRIAGE — QUEUE & PRIORITY (EN/DE/ES/FR)")
    print("=" * 74)
    token_pattern = contract["token_pattern"]
    csv_path = download_csv("tobiasbueck/email-ticket-text-german-classification")

    if csv_path is not None:
        df = pd.read_csv(csv_path)
        cols = {c.lower(): c for c in df.columns}
        subj = df[cols["subject"]].fillna("") if "subject" in cols else ""
        body = df[cols["body"]].fillna("") if "body" in cols else df.iloc[:, 0].fillna("")
        df["text"] = (subj.astype(str) + " " + body.astype(str)).str.strip()
        q_col = cols.get("queue", cols.get("department"))
        p_col = cols.get("priority")
        df = df[["text", q_col, p_col]].dropna().rename(columns={q_col: "queue", p_col: "priority"})
        print(f"Loaded Multilingual Tickets ({len(df):,} rows | {df['queue'].nunique()} queues, {df['priority'].nunique()} priorities)")
    else:
        df = pd.DataFrame({
            "text": [
                "Server outage in production urgent", "Billing invoice question",
                "Passwort zurücksetzen bitte", "Problème de paiement carte refusée",
            ] * 60,
            "queue": ["Technical Support", "Billing and Payments", "IT Support", "Billing and Payments"] * 60,
            "priority": ["high", "low", "medium", "high"] * 60,
        })

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    X_train, X_test = train_df["text"].tolist(), test_df["text"].tolist()

    queue_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(token_pattern=token_pattern, lowercase=True, max_features=20000)),
        ("clf", LogisticRegression(max_iter=1000, C=2.0)),
    ]).fit(X_train, train_df["queue"].astype(str).tolist())

    prio_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(token_pattern=token_pattern, lowercase=True, max_features=20000)),
        ("clf", LogisticRegression(max_iter=1000, C=2.0)),
    ]).fit(X_train, train_df["priority"].astype(str).tolist())

    print(f"Queue    Macro-F1: {f1_score(test_df['queue'].astype(str), queue_pipe.predict(X_test), average='macro'):.4f}")
    print(f"Priority Macro-F1: {f1_score(test_df['priority'].astype(str), prio_pipe.predict(X_test), average='macro'):.4f}")

    q_fix = export_text_clf_onnx(queue_pipe, token_pattern, ARTIFACTS_DIR / "queue.onnx", X_test[:30])
    p_fix = export_text_clf_onnx(prio_pipe, token_pattern, ARTIFACTS_DIR / "priority.onnx", X_test[:30])

    return [
        {
            "text": q["text"],
            "expected_queue": q["expected_label"],
            "expected_queue_prob": q["expected_prob"],
            "expected_priority": p["expected_label"],
            "expected_priority_prob": p["expected_prob"],
        }
        for q, p in zip(q_fix, p_fix)
    ]


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    contract, schema_hash = load_contract()

    run_tfidf_skew_audit(contract["token_pattern"])
    intent_fixtures, bm25_fixtures = train_intent_and_bm25(contract)
    ner_fixtures = train_ner_crf()
    triage_fixtures = train_multilingual_triage(contract)

    out = {
        "schema_version": contract["schema_version"],
        "schema_hash": schema_hash,
        "intent": intent_fixtures,
        "bm25": bm25_fixtures,
        "ner": ner_fixtures,
        "triage": triage_fixtures,
    }
    (ARTIFACTS_DIR / "golden_fixtures.json").write_text(json.dumps(out, indent=2))
    print(f"\nExported unified golden_fixtures.json (schema_hash={schema_hash})")


if __name__ == "__main__":
    main()
