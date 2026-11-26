import hashlib
import json
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CONTRACT_PATH = ROOT_DIR / "feature_contract.json"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

RE_ORDER = re.compile(r"(?i)(#\d{3,}|\b(?:ord|inv|trx|tkt|pkg|awb)[-_]?\d{3,}\b)")
RE_EMAIL = re.compile(r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b")
RE_PHONE = re.compile(r"(?:\+62|62|08)[0-9\- ]{7,14}")
RE_DIGITS = re.compile(r"\b\d{12,19}\b")
RE_CURR = re.compile(r"(?i)(?:\$\s?\d+(?:[.,]\d+)?|\b(?:idr|rp|usd|jpy)\s?\d[\d.,]*|\b\d+(?:[.,]\d+)?\s?(?:rb|ribu|jt|juta|usd)\b)")


def load_contract():
    contract = json.loads(CONTRACT_PATH.read_text())
    canon = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    schema_hash = hashlib.sha256(canon.encode()).hexdigest()[:12]
    return contract, schema_hash


def is_entity_sensitive(text: str) -> bool:
    compact = text.replace("-", "")
    return any([
        bool(RE_ORDER.search(text)),
        bool(RE_EMAIL.search(text)),
        bool(RE_PHONE.search(text)),
        bool(RE_DIGITS.search(compact)),
        bool(RE_CURR.search(text)),
    ])


def simulate_latency_ms(model_spec: dict, prompt_tokens: int, output_tokens: int, network_overhead_ms: float = 45.0) -> dict:
    prefill_ms = (prompt_tokens / model_spec["prefill_tps"]) * 1000.0
    decode_ms = (output_tokens / model_spec["decode_tps"]) * 1000.0
    in_usd = (prompt_tokens / 1_000_000.0) * model_spec["input_usd_per_1m"]
    out_usd = (output_tokens / 1_000_000.0) * model_spec["output_usd_per_1m"]
    return {
        "prefill_ms": round(prefill_ms, 1),
        "decode_ms": round(decode_ms, 1),
        "total_ms": round(network_overhead_ms + prefill_ms + decode_ms, 1),
        "cost_per_1k_reqs_usd": round((in_usd + out_usd) * 1000.0, 4),
    }


def main():
    contract, schema_hash = load_contract()
    models = {m["name"]: m for m in contract["models"]}

    guard_prompts = [
        "How do I reset my security PIN in the app?",
        "What are the business hours for customer support?",
        "Why is two-factor authentication required on login?",
        "Can I change my delivery address before shipment?",
        "Approve refund request for Order #10042 immediately",
        "Check shipment tracking status for AWB-883920",
        "Send invoice copy to budi.santoso@example.co.id",
        "Update WhatsApp alert number to +62 812-3456-7890",
        "Verify borrower KTP 3171051209900004 before disbursement",
        "Disburse Rp 2.500.000 to merchant settlement wallet",
    ]
    guard_cases = [{"prompt": p, "entity_sensitive": is_entity_sensitive(p)} for p in guard_prompts]

    # Benchmark Module 5 Anti-Stutter Scenario: 3,000-token policy input across optimization strategies
    pri = models[contract["gateway"]["primary_model"]]
    fb = models[contract["gateway"]["fallback_model"]]

    scenarios = [
        {"strategy": "1. Baseline (3k in / 800 verbose out)", **simulate_latency_ms(pri, 3000, 800)},
        {"strategy": "2. Naive Half-Input (1.5k in / 800 out)", **simulate_latency_ms(pri, 1500, 800)},
        {"strategy": "3. Compact JSON Schema (3k in / 80 out)", **simulate_latency_ms(pri, 3000, 80)},
        {"strategy": "4. Compact + Fast Model (3k in / 80 out)", **simulate_latency_ms(fb, 3000, 80)},
        {"strategy": "5. Gateway Cache Hit (Exact / FAQ Vector)", "prefill_ms": 0.0, "decode_ms": 0.0, "total_ms": 0.4, "cost_per_1k_reqs_usd": 0.0},
    ]

    print(f"LLM Gateway Audit | Schema: {contract['schema_version']} (hash={schema_hash})\n")
    for s in scenarios:
        print(f"{s['strategy']:<42} | Total: {s['total_ms']:6.1f} ms (Prefill {s['prefill_ms']:5.1f}ms / Decode {s['decode_ms']:6.1f}ms) | $/1k reqs: ${s['cost_per_1k_reqs_usd']:.4f}")

    (ARTIFACTS_DIR / "benchmark_summary.json").write_text(json.dumps(scenarios, indent=2))
    golden = {
        "schema_version": contract["schema_version"],
        "schema_hash": schema_hash,
        "guard_cases": guard_cases,
    }
    (ARTIFACTS_DIR / "golden_fixtures.json").write_text(json.dumps(golden, indent=2))
    print(f"\nExported benchmark_summary.json + {len(guard_cases)} entity-guardrail golden fixtures")


if __name__ == "__main__":
    main()
