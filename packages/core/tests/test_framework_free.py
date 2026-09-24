import subprocess
import sys
import tomllib
from pathlib import Path

HEAVY = ("torch", "transformers", "pydantic", "langgraph", "langchain", "fastapi", "litellm")


def test_core_has_no_runtime_dependencies() -> None:
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text("utf-8"))
    assert pyproject["project"]["dependencies"] == []


def test_importing_core_loads_no_framework() -> None:
    code = f"import sys, medrag_core; print(','.join(m for m in {HEAVY!r} if m in sys.modules))"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == ""
