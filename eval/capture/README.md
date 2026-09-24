# eval/capture

`capture_reference.py` (Step 2) and `capture_step4.py` (chunker and BM25 fixtures) produced the files in `eval/reference/`, `eval/golden/` and the
`tests/fixtures/research_work/` folders by running the original research-work code on fixed
inputs. It is kept for provenance and only needs to be re-run if the reference must be rebuilt.

It runs with the research-work project's own virtualenv (Python 3.10, PyTorch, FAISS) and writes
nothing inside that project. See the module docstring for the exact command.
