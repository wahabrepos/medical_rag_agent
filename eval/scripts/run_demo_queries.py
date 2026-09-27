"""Ask the running API a set of demo questions and keep every full response.

    uv run python eval/scripts/run_demo_queries.py eval/demos/patient_queries.jsonl \
        --api http://127.0.0.1:8010 --out eval/demos/patient_queries.results.jsonl

Each line of the input has at least "id" and "query". Questions are asked one at a
time as free text with include_unverified, so the model's answer is kept even when
the evidence gate withholds it; answered ids are skipped on a re-run.
"""

import argparse
import json
import time
from pathlib import Path

import httpx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("queries", type=Path)
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--key", default="", help="bearer key, if the API needs one")
    args = ap.parse_args()

    queries = [json.loads(line) for line in args.queries.read_text("utf-8").splitlines() if line]
    done = set()
    if args.out.exists():
        done = {json.loads(line)["id"] for line in args.out.read_text("utf-8").splitlines() if line}
    headers = {"Authorization": f"Bearer {args.key}"} if args.key else {}
    with (
        httpx.Client(base_url=args.api, timeout=300, headers=headers) as client,
        args.out.open("a", encoding="utf-8") as out,
    ):
        for q in queries:
            if q["id"] in done:
                continue
            body = {"question": q["query"], "answer_format": "free", "include_unverified": True}
            for _attempt in range(3):
                started = time.monotonic()
                response = client.post("/v1/ask", json=body)
                if response.status_code == 200:
                    break
                print(f"{q['id']}: HTTP {response.status_code}, retrying", flush=True)
                time.sleep(20)
            response.raise_for_status()
            answer = response.json()
            row = {**q, "seconds": round(time.monotonic() - started, 1), "response": answer}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            model = (answer["model_answer"] or "")[:70]
            print(f"{q['id']:>2} {answer['answer'][:30]!r:32} model: {model!r}", flush=True)


if __name__ == "__main__":
    main()
