import base64
import json
import random
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def post_json(url: str, payload: dict) -> int:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        resp.read()
        return resp.status


def post_bytes(url: str, raw: bytes, content_type: str = "application/octet-stream") -> int:
    req = urllib.request.Request(url, data=raw, headers={"Content-Type": content_type}, method="POST")
    with urllib.request.urlopen(req, timeout=8) as resp:
        resp.read()
        return resp.status


def load_json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def extract_01_inputs(gf: dict) -> list[dict]:
    cases = gf.get("cases") or gf.get("golden_cases") or []
    out = [c.get("input") or c.get("features") for c in cases if isinstance(c, dict)]
    out = [x for x in out if isinstance(x, dict)]
    return out or [{"distance_km": 420.0, "weight_grams": 1200.0, "seller_late_rate": 0.18, "is_weekend": 1, "freight_ratio": 0.22}]


def collect_strings(node, keys=("text", "input", "query", "utterance")) -> list[str]:
    found = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k in keys and isinstance(v, str) and len(v.strip()) > 3:
                found.append(v)
            else:
                found.extend(collect_strings(v, keys))
    elif isinstance(node, list):
        for item in node:
            found.extend(collect_strings(item, keys))
    return found


def extract_media_04(gf: dict) -> tuple[list[bytes], list[bytes]]:
    pngs, wavs = [], []

    def walk(node):
        if isinstance(node, dict):
            for k, val in node.items():
                if isinstance(val, str) and len(val) > 64:
                    if "png" in k or "image" in k:
                        try:
                            pngs.append(base64.b64decode(val))
                        except Exception:
                            pass
                    elif "wav" in k or "audio" in k:
                        try:
                            wavs.append(base64.b64decode(val))
                        except Exception:
                            pass
                else:
                    walk(val)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(gf)
    return pngs, wavs


def build_static_mp4() -> bytes | None:
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "lavfi", "-i", "color=c=navy:s=128x128:d=3:r=5",
                    "-f", "lavfi", "-i", "sine=frequency=220:duration=3",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    tmp.name, "-loglevel", "error",
                ],
                check=True,
                timeout=8,
            )
            return Path(tmp.name).read_bytes()
    except Exception:
        return None


def main():
    sla_inputs = extract_01_inputs(load_json("01-delivery-sla-breach/artifacts/golden_fixtures.json"))
    nlp_texts = collect_strings(load_json("02-classic-nlp/artifacts/golden_fixtures.json")) + [
        "Please call Budi at 081234567890 or email budi.santoso@example.com for refund",
        "My KTP NIK is 3171234567890001 and my parcel is missing",
    ]
    trf_texts = collect_strings(load_json("03-transformer-onnx/artifacts/golden_fixtures.json")) + [
        "Where can I track my courier shipment?",
        "Charged twice on my credit card invoice, need refund",
        "Cannot reset my two-factor password on mobile app",
    ]
    pngs, wavs = extract_media_04(load_json("04-multimodal-ai/artifacts/golden_fixtures.json"))
    sample_mp4 = build_static_mp4()

    rag_cases = load_json("06-production-rag/artifacts/golden_fixtures.json")["golden_cases"]
    rag_pool = [{"tenant_id": c["tenant_id"], "query": c["query"]} for c in rag_cases] + [
        {"tenant_id": "LEGAL", "query": "sovereign maritime arbitration treaty clause"},
    ]
    agent_cases = load_json("07-agents-mcp-evals/artifacts/golden_fixtures.json")["golden_cases"]

    print("Streaming enriched golden-fixture traffic across :8081 -> :8087 (Ctrl+C to stop)...")
    tick = 0
    counts = {f"0{i}": 0 for i in range(1, 8)}

    try:
        while True:
            tick += 1
            # 01 (:8081)
            try:
                post_json("http://localhost:8081/predict", random.choice(sla_inputs))
                counts["01"] += 1
            except urllib.error.URLError:
                pass

            # 02 (:8082)
            try:
                txt = random.choice(nlp_texts)
                route = random.choice(("/intent", "/triage", "/redact", "/search"))
                post_json(f"http://localhost:8082{route}", {"query": txt} if route == "/search" else {"text": txt})
                counts["02"] += 1
            except urllib.error.URLError:
                pass

            # 03 (:8083)
            try:
                prec = "int8" if tick % 2 == 0 else "fp32"
                post_json("http://localhost:8083/predict", {"text": random.choice(trf_texts), "precision": prec})
                counts["03"] += 1
            except urllib.error.URLError:
                pass

            # 04 (:8084)
            try:
                if sample_mp4 and tick % 8 == 0:
                    ep = "/ekyc/video" if tick % 16 == 0 else "/unboxing/video"
                    post_bytes(f"http://localhost:8084{ep}", sample_mp4, "video/mp4")
                elif pngs and tick % 2 == 0:
                    mode = "ekyc" if tick % 4 == 0 else "unboxing"
                    post_bytes(f"http://localhost:8084/image/classify?mode={mode}", random.choice(pngs), "image/png")
                elif wavs:
                    post_bytes("http://localhost:8084/audio/vad", random.choice(wavs), "audio/wav")
                counts["04"] += 1
            except urllib.error.URLError:
                pass

            # 05 (:8085) — exercises exact_sha256, semantic_cosine (temp=0.2), entity guard, and live 429 failover
            try:
                gw_choice = tick % 4
                if gw_choice == 0:
                    gw_req = {"tenant_id": "ORDER", "temperature": 0.0, "messages": [{"role": "user", "content": "What are your standard support hours?"}]}
                elif gw_choice == 1:
                    gw_req = {"tenant_id": "ORDER", "temperature": 0.2, "messages": [{"role": "user", "content": random.choice([
                        "How many days is the return window for electronics?",
                        "What is the electronics return window in days?",
                    ])}]}
                elif gw_choice == 2:
                    gw_req = {"tenant_id": "ORDER", "temperature": 0.2, "messages": [{"role": "user", "content": f"Status of ORD-{1000 + (tick % 50)} paid 180000 IDR"}]}
                else:
                    gw_req = {"tenant_id": "REFUND", "temperature": 0.3, "fault_inject": "primary_429", "messages": [{"role": "user", "content": f"Failover probe #{tick}"}]}
                post_json("http://localhost:8085/v1/chat/completions", gw_req)
                counts["05"] += 1
            except urllib.error.URLError:
                pass

            # 06 (:8086) — queries + periodic live policy ingest
            try:
                post_json("http://localhost:8086/v1/rag/query", random.choice(rag_pool))
                if tick % 25 == 0:
                    post_json("http://localhost:8086/v1/rag/ingest", {
                        "chunk_id": f"POL-LIVE-{tick % 5:02d}#0",
                        "tenant_id": "ORDER",
                        "policy_code": "POL-LIVE",
                        "header": "Live Ops SLA Bulletin",
                        "content": "Orders delayed past 48 hours qualify for priority courier rerouting.",
                    })
                counts["06"] += 1
            except urllib.error.URLError:
                pass

            # 07 (:8087)
            try:
                ac = random.choice(agent_cases)
                post_json("http://localhost:8087/v1/agent/run", {
                    "session_tenant": ac["session_tenant"],
                    "user_id": ac["user_id"],
                    "user_query": ac["user_query"],
                    "requested_refund_idr": ac["requested_refund_idr"],
                    "simulate_duplicate_retry": ac["simulate_duplicate_retry"],
                })
                counts["07"] += 1
            except urllib.error.URLError:
                pass

            if tick % 10 == 0:
                print(f"[round {tick}] " + " ".join(f"{k}:{v}" for k, v in counts.items()))
            time.sleep(0.30)
    except KeyboardInterrupt:
        print(f"\nStopped. Final counts: {counts}")


if __name__ == "__main__":
    main()
