"""End-to-end smoke test of the Compose stack through its web entry point (Caddy).

    uv run python eval/scripts/smoke_stack.py --url http://127.0.0.1:8090

Checks, in order: the UI page is served; /healthz and /readyz answer through the
proxy; one question gets a complete answer (the gate may show or withhold it; both are
valid); the streamed endpoint delivers progress and the answer over SSE; feedback is
stored and the run can be read back. Costs two LLM questions (about EUR 0.001).
Exits non-zero at the first failed check.
"""

import argparse
import json
import sys
import time

import httpx

QUESTION = "Does increased use of private health care reduce the demand for NHS care?"
ANSWER_FIELDS = {"run_id", "answer", "evidence", "citations", "iterations", "model", "disclaimer"}


def check(ok: bool, message: str) -> None:
    print(("PASS " if ok else "FAIL ") + message, flush=True)
    if not ok:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--key", default="", help="bearer key, if API_KEYS is set")
    args = ap.parse_args()
    headers = {"Authorization": f"Bearer {args.key}"} if args.key else {}

    with httpx.Client(base_url=args.url, timeout=300, headers=headers) as client:
        page = client.get("/")
        check(page.status_code == 200 and "MedRAG" in page.text, "web UI served at /")
        check(client.get("/healthz").json().get("status") == "ok", "API /healthz through the proxy")
        ready = client.get("/readyz")
        check(ready.status_code == 200, f"API /readyz: {ready.text}")
        grounding = ready.json()["checks"].get("grounding", "")
        print(f"     verifier: {ready.json()['checks'].get('nli')} ({grounding})")

        started = time.monotonic()
        response = client.post("/v1/ask", json={"question": QUESTION, "answer_format": "yes_no"})
        check(response.status_code == 200, f"POST /v1/ask ({response.status_code})")
        answer = response.json()
        check(answer.keys() >= ANSWER_FIELDS, "answer has the documented fields")
        shown = answer["answer"] != "insufficient evidence"
        print(
            f"     {'shown' if shown else 'withheld'}: {answer['answer']!r}; "
            f"{len(answer['evidence']['statements'])} statements, "
            f"{len(answer['citations'])} sources, {time.monotonic() - started:.1f} s"
        )
        check(len(answer["citations"]) > 0, "sources were retrieved")

        events: list[str] = []
        body = {"question": QUESTION, "answer_format": "yes_no"}
        with client.stream("POST", "/v1/ask/stream", json=body) as stream:
            for line in stream.iter_lines():
                if line.startswith("event: "):
                    events.append(line.removeprefix("event: "))
        check(
            "retrieved" in events and events[-1] == "answer",
            f"SSE stream through the proxy: {', '.join(events)}",
        )

        run_id = answer["run_id"]
        feedback = client.post(
            f"/v1/runs/{run_id}/feedback", json={"rating": 1, "comment": "smoke test"}
        )
        check(feedback.status_code == 201, "feedback stored")
        stored = client.get(f"/v1/runs/{run_id}")
        check(stored.status_code == 200 and stored.json()["run_id"] == run_id, "run read back")
    print(json.dumps({"smoke": "passed", "url": args.url}))


if __name__ == "__main__":
    main()
