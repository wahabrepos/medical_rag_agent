# eval/golden

`golden_150.jsonl`: the fixed pull-request evaluation set, 75 MedQA + 75 PubMedQA questions
drawn with seed 42 from the research-work evaluation sets. Each row holds the question exactly
as the research work sent it, the correct answer, and the research-work predictions for every
system (including the Self-MedRAG + Mistral-small rationale).

The full sets (1,000 + 890) are rebuilt on demand with `eval/scripts/fetch_eval_sets.py`.
