import json
import random
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAG_URL = "http://localhost:8086/v1/rag/query"

ABSTAIN_PROBES = [
    {"tenant_id": "ORDER", "query": "quantum chromodynamics lattice gauge theory"},
    {"tenant_id": "ACCOUNT", "query": "mars rover orbital insertion telemetry"},
]


def post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    fixtures_path = ROOT / "06-production-rag" / "artifacts" / "golden_fixtures.json"
    cases = json.loads(fixtures_path.read_text())["golden_cases"]
    pool = [{"tenant_id": c["tenant_id"], "query": c["query"]} for c in cases] + ABSTAIN_PROBES

    print(f"Replaying {len(cases)} golden RAG fixtures + {len(ABSTAIN_PROBES)} abstention probes -> {RAG_URL} (Ctrl+C to stop)")
    sent = 0
    try:
        while True:
            sample = random.choice(pool)
            out = post_json(RAG_URL, sample)
            sent += 1
            if sent % 10 == 0:
                print(f"[{sent}] last={sample['tenant_id']} abstained={out['abstained']} verified={out['citation_verified']}")
            time.sleep(0.35)
    except KeyboardInterrupt:
        print(f"\nStopped after {sent} requests.")


if __name__ == "__main__":
    main()