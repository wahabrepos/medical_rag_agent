"""Write a Markdown report of demo questions answered by run_demo_queries.py.

    uv run python eval/scripts/report_demo_queries.py eval/demos/patient_queries \
        --title "Patient-style questions"

Reads <stem>.results.jsonl (the API's responses) and <stem>.grades.json (a human
judgement per question: match / partial / different / declined, with a note) and
writes <stem>.md with every question, the model's answer, what the evidence gate
showed, the checked statements and the sources.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

VERDICTS = {
    "match": "✅ match",
    "partial": "🟡 partial",
    "different": "❌ different",
    "declined": "⚪ declined",
}


def status(response: dict) -> str:
    return "shown" if response["answer"] != "insufficient evidence" else "withheld"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stem", type=Path)
    ap.add_argument("--title", default="Demo questions")
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in args.stem.with_suffix(".results.jsonl").read_text("utf-8").splitlines()
        if line
    ]
    grades = json.loads(args.stem.with_suffix(".grades.json").read_text("utf-8"))
    counts = Counter(grades[str(r["id"])]["verdict"] for r in rows)
    shown = sum(status(r["response"]) == "shown" for r in rows)
    lines = [
        f"# {args.title}",
        "",
        f"{len(rows)} questions asked one by one through the running system (API, "
        f"{rows[0]['response']['model']}, MiniCheck verifier), as free text with the model's "
        "own answer requested. Verdicts compare that answer with the expected diagnosis and "
        "were judged by hand.",
        "",
        f"- Evidence gate: **{shown} shown**, {len(rows) - shown} withheld as "
        '"insufficient evidence"',
        "- Model's answer vs expected: "
        + ", ".join(f"{VERDICTS[v]} {counts.get(v, 0)}" for v in VERDICTS),
        "",
        "| # | Domain | Patient's words | Expected | Model's answer | Gate | Verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        a = r["response"]
        g = grades[str(r["id"])]
        answer = (a["model_answer"] or a["answer"]).replace("|", "/")
        lines.append(
            f"| {r['id']} | {r['domain']} | {r['query']} | {r['expected']} | {answer} "
            f"| {status(a)} | {VERDICTS[g['verdict']]} ({g['note']}) |"
        )
    lines += ["", "## Answers in full", ""]
    for r in rows:
        a = r["response"]
        lines += [
            f"### {r['id']}. {r['domain']}",
            "",
            f"> {r['query']}",
            "",
            f"- Expected: {r['expected']} ({r['technical_term']})",
            f"- Shown to the user: **{a['answer']}**",
            f"- Model's own answer: {a['model_answer'] or a['answer']}",
            f"- Evidence: {a['evidence']['status']}; {a['iterations']} attempt(s); "
            f"{r['seconds']} s",
            f"- Verdict: {VERDICTS[grades[str(r['id'])]['verdict']]} "
            f"({grades[str(r['id'])]['note']})",
            "",
        ]
        for s in a["evidence"]["statements"]:
            if s["from_question"]:
                tag = "from the question"
            elif s["supported"] and s["supporting_citation"] is not None:
                tag = f"source [{s['supporting_citation'] + 1}], support {s['support']:.2f}"
            else:
                tag = f"not found in the sources, support {s['support']:.2f}"
            lines.append(f"  - {s['text']} ({tag})")
            if s["quote"]:
                lines.append(f'    > "{s["quote"]}"')
        if a["citations"]:
            lines.append("")
            lines.append("  Retrieved sources:")
            for i, c in enumerate(a["citations"], start=1):
                link = f"[{c['title']}]({c['url']})" if c["url"] else c["title"]
                lines.append(f"  {i}. {link}")
        lines.append("")
    out = args.stem.with_suffix(".md")
    out.write_text("\n".join(lines).rstrip("\n") + "\n", "utf-8")
    print(f"{out}: {len(rows)} questions, {shown} shown, {dict(counts)}")


if __name__ == "__main__":
    main()
