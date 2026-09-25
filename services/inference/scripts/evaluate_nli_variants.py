"""Compare NLI model variants against the research work's PyTorch outputs.

Produced reports/nli_variants_x86_zen4.json, which is why the service uses fp32.

Evaluates fp32 ONNX, the published int8 exports for this CPU and our own dynamic
int8 quantisations on the captured fixture (250 real pairs), against the ADR-0005
bar: probabilities within 0.02 (98th percentile) of the reference and the same
decision at theta = 0.7 for at least 98% of pairs, for every label. Also checks that
the research-work support scores are reproduced and measures speed.

Run it on a machine with the target CPU architecture (int8 kernels are CPU-specific).
Needs onnxruntime, onnx, tokenizers, huggingface-hub, numpy and nli.py next to it:

    python evaluate_nli_variants.py nli_cases.json --out report.json --keep-best best.onnx
"""

import argparse
import json
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from nli import LABELS, NLI_MODEL_ID, NLI_REVISION, DebertaNli

PUBLISHED = {
    "published_quint8_avx2": "onnx/model_quint8_avx2.onnx",
    "published_qint8_avx512": "onnx/model_qint8_avx512.onnx",
    "published_qint8_avx512_vnni": "onnx/model_qint8_avx512_vnni.onnx",
    "published_qint8_arm64": "onnx/model_qint8_arm64.onnx",
}
THETA = 0.7
BUDGET_PAIRS, BUDGET_SECONDS = 25, 2.0


def quantise(src: str, out: Path, *, per_channel: bool, reduce_range: bool) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        src,
        str(out),
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        reduce_range=reduce_range,
    )
    return out


def evaluate(
    nli: DebertaNli,
    path: str,
    pairs: list[tuple[str, str]],
    reference: np.ndarray,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {"file": path, "size_mb": round(Path(path).stat().st_size / 1e6, 1)}
    for threads in (2, 0):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        nli.session = ort.InferenceSession(
            path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        nli._input_names = {i.name for i in nli.session.get_inputs()}
        nli.probabilities(pairs[:4])
        started = time.perf_counter()
        probs = nli.probabilities(pairs)
        elapsed = time.perf_counter() - started
        started = time.perf_counter()
        nli.probabilities(pairs[:BUDGET_PAIRS])
        budget = time.perf_counter() - started
        key = "threads_2" if threads == 2 else "threads_all"
        result[key] = {
            "pairs_per_second": round(len(pairs) / elapsed, 2),
            "seconds_for_25_pairs": round(budget, 3),
        }

    diff = np.abs(probs - reference)
    agreement = {
        label: float(np.mean((probs[:, i] >= THETA) == (reference[:, i] >= THETA)))
        for i, label in enumerate(LABELS)
    }
    col = LABELS.index("neutral")  # the research work's "support" column
    reproduced, n = 0, 0
    offset = 0
    for case in cases:
        k = len(case["passages"])
        rows = probs[offset : offset + k * len(case["rationale"])]
        offset += k * len(case["rationale"])
        supported = sum(
            rows[i * k : (i + 1) * k, col].max() >= THETA for i in range(len(case["rationale"]))
        )
        reproduced += int(supported / len(case["rationale"]) == case["research_work_support_score"])
        n += 1
    result.update(
        {
            "max_abs_diff": round(float(diff.max()), 4),
            "p98_abs_diff": round(float(np.quantile(diff, 0.98)), 4),
            "mean_abs_diff": round(float(diff.mean()), 5),
            "decision_agreement": {k: round(v, 4) for k, v in agreement.items()},
            "support_scores_reproduced": f"{reproduced}/{n}",
        }
    )
    result["passes_accuracy_bar"] = (
        result["p98_abs_diff"] <= 0.02 and min(agreement.values()) >= 0.98
    )
    result["meets_speed_budget_2_threads"] = (
        result["threads_2"]["seconds_for_25_pairs"] <= BUDGET_SECONDS
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fixture", type=Path)
    ap.add_argument("--out", type=Path, default=Path("nli_variants_report.json"))
    ap.add_argument("--keep-best", type=Path, default=None)
    args = ap.parse_args()

    fixture = json.loads(args.fixture.read_text("utf-8"))
    cases = fixture["cases"]
    pairs = [(p, s) for c in cases for s in c["rationale"] for p in c["passages"]]
    reference = np.array([row for c in cases for row in c["probabilities"]], dtype=np.float32)
    nli = DebertaNli(cache_size=0)
    fp32_path = hf_hub_download(NLI_MODEL_ID, "onnx/model.onnx", revision=NLI_REVISION)

    variants: dict[str, str] = {"fp32": fp32_path}
    for name, file in PUBLISHED.items():
        if "arm64" in name and platform.machine() not in ("aarch64", "arm64"):
            continue
        variants[name] = hf_hub_download(NLI_MODEL_ID, file, revision=NLI_REVISION)
    work = Path("quantised")
    work.mkdir(exist_ok=True)
    for per_channel in (True, False):
        for reduce_range in (False, True):
            name = f"own_qint8_{'per_channel' if per_channel else 'per_tensor'}" + (
                "_reduce_range" if reduce_range else ""
            )
            started = time.perf_counter()
            path = quantise(
                fp32_path, work / f"{name}.onnx", per_channel=per_channel, reduce_range=reduce_range
            )
            print(f"{name}: quantised in {time.perf_counter() - started:.0f}s", flush=True)
            variants[name] = str(path)

    report: dict[str, Any] = {
        "cpu": platform.processor() or platform.machine(),
        "onnxruntime": ort.__version__,
        "pairs": len(pairs),
        "variants": {},
    }
    for name, path in variants.items():
        report["variants"][name] = evaluate(nli, path, pairs, reference, cases)
        print(name, json.dumps(report["variants"][name]), flush=True)

    passing = [
        (v["threads_2"]["seconds_for_25_pairs"], name)
        for name, v in report["variants"].items()
        if name != "fp32" and v["passes_accuracy_bar"]
    ]
    report["best_int8"] = min(passing)[1] if passing else None
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    if args.keep_best and report["best_int8"]:
        shutil.copyfile(variants[report["best_int8"]], args.keep_best)
    print("best int8:", report["best_int8"])


if __name__ == "__main__":
    main()
