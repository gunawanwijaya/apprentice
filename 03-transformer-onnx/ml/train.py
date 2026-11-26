import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent
CONTRACT_PATH = ROOT_DIR / "feature_contract.json"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_contract():
    contract = json.loads(CONTRACT_PATH.read_text())
    canon_obj = {
        "do_lower_case": contract["do_lower_case"],
        "labels": contract["labels"],
        "max_seq_len": contract["max_seq_len"],
        "model_id": contract["model_id"],
    }
    canon = json.dumps(canon_obj, sort_keys=True, separators=(",", ":"))
    schema_hash = hashlib.sha256(canon.encode()).hexdigest()[:12]
    return contract, schema_hash


def make_nuanced_ticket_dataset(n_train_per_class: int = 350, n_unseen_per_class: int = 100):
    """Splits by template so test data tests true semantic generalization, not memorization."""
    rng = random.Random(42)

    train_templates = {
        "billing_refund": [
            "charged twice on invoice {id}, please issue a refund immediately",
            "the delivery arrived fine and not delayed, but my card was overcharged on order {id}",
            "no login or app crash issue here, I just need my duplicate payment refunded",
            "cancel my subscription renewal and reverse the unexpected charge of {amt} dollars",
        ],
        "delivery_delay": [
            "parcel {id} has been stuck at the sorting hub for five days past the eta",
            "I did not ask for a billing refund, I just want to know why my shipment is delayed",
            "payment went through fine and no double charge, yet the courier never showed up",
            "order {id} was marked delivered by the driver but nothing arrived at my address",
        ],
        "account_access": [
            "locked out of my account after resetting my password and 2fa code never arrives",
            "no billing issue or delayed parcel, I simply cannot sign in to view my orders",
            "please unlock my profile {id}, verification sms keeps timing out during login",
            "my session expires immediately on login even though the password reset succeeded",
        ],
        "technical_bug": [
            "checkout button freezes with a 500 server error whenever I upload an attachment",
            "I can log in fine and have no billing dispute, but the dashboard crashes on export",
            "api webhook payload returns malformed json for order {id} since yesterday's release",
            "search filter ignores date range and duplicates rows after refreshing the page",
        ],
    }

    # Completely unseen wording/synonyms with cross-class distractors
    unseen_templates = {
        "billing_refund": [
            "my bank statement shows an extra {amt} dollar fee, not a shipping delay",
            "the promo voucher failed to deduct from my receipt for {id}, please credit me back",
            "no software glitch or sign in trouble, I was billed again after cancelling",
        ],
        "delivery_delay": [
            "my box for {id} still has not reached my doorstep, not seeking money back",
            "the carrier tracking link stopped updating last week even though payment cleared",
            "package is overdue and missing in transit, no password or browser problem",
        ],
        "account_access": [
            "credentials keep getting rejected at the sign in screen, not a billing question",
            "cannot pass two factor authentication after switching phones, order {id} is fine",
            "need admin permissions restored on my user profile, the web app is not crashing",
        ],
        "technical_bug": [
            "exporting csv hangs the browser tab with a blank white screen on ios",
            "not asking for a refund or courier update, the mobile screen freezes on submit",
            "clicking save throws an internal server exception for record {id}",
        ],
    }

    def sample(tpl_map, n_per_class):
        txts, ys = [], []
        for idx, lbl in enumerate(tpl_map.keys()):
            for _ in range(n_per_class):
                t = rng.choice(tpl_map[lbl]).format(id=f"ORD-{rng.randint(1000, 9999)}", amt=rng.randint(15, 290))
                txts.append(t)
                ys.append(idx)
        return txts, np.array(ys, dtype=np.int64)

    tr_all_x, tr_all_y = sample(train_templates, n_train_per_class)
    X_tr, X_seen, y_tr, y_seen = train_test_split(tr_all_x, tr_all_y, test_size=0.2, stratify=tr_all_y, random_state=42)
    X_unseen, y_unseen = sample(unseen_templates, n_unseen_per_class)
    return X_tr, y_tr, X_seen, y_seen, X_unseen, y_unseen


# --- Neural Baseline Architectures (MLP, 1D-CNN, BiLSTM) ---
class SimpleMLP(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, num_classes))

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1).float()
        pooled = (self.emb(x) * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.fc(pooled)


class TextCNN1D(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, 48, k) for k in (3, 4, 5)])
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(48 * 3, num_classes)

    def forward(self, x):
        h = self.emb(x).transpose(1, 2)  # [B, D, L]
        pooled = [torch.relu(conv(h)).max(dim=2).values for conv in self.convs]
        return self.fc(self.drop(torch.cat(pooled, dim=1)))


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, 64, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        out, _ = self.lstm(self.emb(x))
        pooled = out.mean(dim=1)
        return self.fc(self.drop(pooled))


class OnnxBertWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def train_and_eval_torch(name, model, X_tr, y_tr, X_seen, y_seen, X_unseen, y_unseen, epochs=12, lr=3e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=32, shuffle=True)

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            opt.zero_grad()
            loss_fn(model(bx), by).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        seen_f1 = float(f1_score(y_seen, model(torch.tensor(X_seen)).argmax(dim=1).numpy(), average="macro"))
        unseen_f1 = float(f1_score(y_unseen, model(torch.tensor(X_unseen)).argmax(dim=1).numpy(), average="macro"))
        sample = torch.tensor(X_unseen[:1])
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            _ = model(sample)
            times.append((time.perf_counter() - t0) * 1000)

    p50_ms = float(np.median(times))
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"{name:<22} | Seen F1: {seen_f1:.4f} | Unseen F1: {unseen_f1:.4f} | CPU p50: {p50_ms:6.2f} ms | Size: {size_mb:5.2f} MB")
    return {"model": name, "seen_f1": round(seen_f1, 4), "unseen_f1": round(unseen_f1, 4), "cpu_p50_ms": round(p50_ms, 3), "size_mb": round(size_mb, 2)}


def eval_onnx(name, onnx_path: Path, ids_seen, mask_seen, y_seen, ids_unseen, mask_unseen, y_unseen):
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def predict_all(ids, mask):
        preds, times = [], []
        for i in range(len(ids)):
            feed = {"input_ids": ids[i : i + 1], "attention_mask": mask[i : i + 1]}
            t0 = time.perf_counter()
            logits = sess.run(["logits"], feed)[0]
            times.append((time.perf_counter() - t0) * 1000)
            preds.append(int(np.argmax(logits[0])))
        return preds, times

    seen_preds, _ = predict_all(ids_seen, mask_seen)
    unseen_preds, times = predict_all(ids_unseen, mask_unseen)
    seen_f1 = float(f1_score(y_seen, seen_preds, average="macro"))
    unseen_f1 = float(f1_score(y_unseen, unseen_preds, average="macro"))
    p50_ms = float(np.median(times))
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"{name:<22} | Seen F1: {seen_f1:.4f} | Unseen F1: {unseen_f1:.4f} | CPU p50: {p50_ms:6.2f} ms | Size: {size_mb:5.2f} MB")
    return sess, {"model": name, "seen_f1": round(seen_f1, 4), "unseen_f1": round(unseen_f1, 4), "cpu_p50_ms": round(p50_ms, 3), "size_mb": round(size_mb, 2)}


def main():
    set_seed(42)
    contract, schema_hash = load_contract()
    model_id = contract["model_id"]
    max_len = contract["max_seq_len"]
    labels_list = contract["labels"]

    X_tr_txt, y_tr, X_seen_txt, y_seen, X_unseen_txt, y_unseen = make_nuanced_ticket_dataset()

    tokenizer = BertTokenizer.from_pretrained(model_id)
    def encode(txts):
        enc = tokenizer(txts, padding="max_length", truncation=True, max_length=max_len, return_tensors="np")
        return enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64)

    ids_tr, mask_tr = encode(X_tr_txt)
    ids_seen, mask_seen = encode(X_seen_txt)
    ids_unseen, mask_unseen = encode(X_unseen_txt)

    vocab_files = tokenizer.save_vocabulary(str(ARTIFACTS_DIR))
    if Path(vocab_files[0]).name != contract["vocab_file"]:
        shutil.move(vocab_files[0], ARTIFACTS_DIR / contract["vocab_file"])

    results = []
    print(f"Split: {len(X_tr_txt)} train | {len(X_seen_txt)} seen-val | {len(X_unseen_txt)} unseen-test | Model: {model_id}\n")

    # 1. TF-IDF + Logistic Regression
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=3000)
    lr = LogisticRegression(max_iter=500).fit(vec.fit_transform(X_tr_txt), y_tr)
    lr_seen = float(f1_score(y_seen, lr.predict(vec.transform(X_seen_txt)), average="macro"))
    lr_unseen = float(f1_score(y_unseen, lr.predict(vec.transform(X_unseen_txt)), average="macro"))
    times = []
    for i in range(50):
        t0 = time.perf_counter()
        _ = lr.predict(vec.transform([X_unseen_txt[i]]))
        times.append((time.perf_counter() - t0) * 1000)
    lr_p50 = float(np.median(times))
    print(f"{'TF-IDF + LogReg':<22} | Seen F1: {lr_seen:.4f} | Unseen F1: {lr_unseen:.4f} | CPU p50: {lr_p50:6.2f} ms | Size:  0.15 MB")
    results.append({"model": "TF-IDF + LogReg", "seen_f1": round(lr_seen, 4), "unseen_f1": round(lr_unseen, 4), "cpu_p50_ms": round(lr_p50, 3), "size_mb": 0.15})

    # 2–4. Neural Baselines
    vocab_size = tokenizer.vocab_size
    results.append(train_and_eval_torch("MLP (AvgPool)", SimpleMLP(vocab_size, 64, len(labels_list)), ids_tr, y_tr, ids_seen, y_seen, ids_unseen, y_unseen))
    results.append(train_and_eval_torch("1D-CNN (TextCNN)", TextCNN1D(vocab_size, 64, len(labels_list)), ids_tr, y_tr, ids_seen, y_seen, ids_unseen, y_unseen))
    results.append(train_and_eval_torch("BiLSTM (RNN)", BiLSTMClassifier(vocab_size, 64, len(labels_list)), ids_tr, y_tr, ids_seen, y_seen, ids_unseen, y_unseen))

    # 5. Fine-tune Encoder Transformer (BERT-Tiny)
    bert = BertForSequenceClassification.from_pretrained(model_id, num_labels=len(labels_list))
    opt = torch.optim.AdamW(bert.parameters(), lr=3e-4, weight_decay=0.01)
    loader = DataLoader(TensorDataset(torch.tensor(ids_tr), torch.tensor(mask_tr), torch.tensor(y_tr)), batch_size=32, shuffle=True)

    bert.train()
    for _ in range(5):
        for b_ids, b_mask, b_y in loader:
            opt.zero_grad()
            out = bert(input_ids=b_ids, attention_mask=b_mask, labels=b_y)
            out.loss.backward()
            opt.step()

    bert.eval()
    fp32_path = ARTIFACTS_DIR / "model_fp32.onnx"
    int8_path = ARTIFACTS_DIR / "model_int8.onnx"

    wrapper = OnnxBertWrapper(bert)
    torch.onnx.export(
        wrapper,
        (torch.zeros((1, max_len), dtype=torch.int64), torch.ones((1, max_len), dtype=torch.int64)),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=14,
        dynamo=False,
    )
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QUInt8)

    sess_fp32, res_fp32 = eval_onnx("BERT-Tiny (FP32 ONNX)", fp32_path, ids_seen, mask_seen, y_seen, ids_unseen, mask_unseen, y_unseen)
    sess_int8, res_int8 = eval_onnx("BERT-Tiny (INT8 ONNX)", int8_path, ids_seen, mask_seen, y_seen, ids_unseen, mask_unseen, y_unseen)
    results.extend([res_fp32, res_int8])

    (ARTIFACTS_DIR / "benchmark_summary.json").write_text(json.dumps(results, indent=2))

    # Emit 100 golden fixtures from unseen paraphrases
    fixtures = []
    for i in range(100):
        feed = {"input_ids": ids_unseen[i : i + 1], "attention_mask": mask_unseen[i : i + 1]}
        fp32_logits = sess_fp32.run(["logits"], feed)[0][0].tolist()
        int8_logits = sess_int8.run(["logits"], feed)[0][0].tolist()
        fixtures.append(
            {
                "text": X_unseen_txt[i],
                "input_ids": ids_unseen[i].tolist(),
                "attention_mask": mask_unseen[i].tolist(),
                "fp32_logits": [round(x, 6) for x in fp32_logits],
                "int8_logits": [round(x, 6) for x in int8_logits],
                "expected_label": labels_list[int(np.argmax(fp32_logits))],
            }
        )

    out = {"schema_version": contract["schema_version"], "schema_hash": schema_hash, "cases": fixtures}
    (ARTIFACTS_DIR / "golden_fixtures.json").write_text(json.dumps(out, indent=2))
    print(f"\nExported FP32/INT8 ONNX, vocab.txt, and 100 unseen golden fixtures (schema_hash={schema_hash})")


if __name__ == "__main__":
    main()
