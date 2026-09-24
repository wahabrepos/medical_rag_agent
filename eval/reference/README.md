# eval/reference

Reference data captured from the original research-work code (Step 2). The migrated
system is measured against it; none of these files should be edited by hand.

| File | Contents |
|---|---|
| `eval_sets_manifest.json` | The exact questions evaluated: 1,000 MedQA + 890 PubMedQA, each with its source-split row, a SHA-256 of the formatted question and the correct answer |
| `predictions/<system>.jsonl` | Per-question predictions of the seven research-work systems, in manifest order |
| `metrics.json` | Stored metrics per system and dataset (each recomputed from the predictions with the original evaluator) |
| `retrieval_200.jsonl` | BM25 top-10, dense top-10 and fused top-5 chunks for 200 questions, for the retrieval parity check |
| `parity_settings.json` | Generator, retrieval, verification and loop settings the parity build must reproduce, plus the accuracy targets |

Parity target (Self-MedRAG + Mistral-small): **MedQA 71.30%, PubMedQA 75.96%**, tolerance ±2 points.

Rebuild the full question sets (not committed) with:

```bash
uv run --env-file .env --with datasets python eval/scripts/fetch_eval_sets.py
```
