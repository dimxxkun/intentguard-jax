"""Fresh-process eager/JIT comparisons. Workers synchronize asynchronous JAX work."""

import subprocess
import sys
from pathlib import Path

import numpy as np

from .io import read_json, write_json


def benchmark(bundle, output, repeats=20, cpu_threads=8):
    if repeats < 1 or cpu_threads < 1:
        raise ValueError("Repetitions and CPU threads must be positive")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Benchmark output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for batch in (1, 8, 32):
        reference = None
        for mode in ("eager", "jit"):
            destination = output / f"{mode}-batch-{batch}"
            command = [
                sys.executable,
                "-m",
                "intentguard.benchmark_worker",
                "--bundle",
                str(Path(bundle).resolve()),
                "--output",
                str(destination),
                "--mode",
                mode,
                "--batch",
                str(batch),
                "--repeats",
                str(repeats),
                "--cpu-threads",
                str(cpu_threads),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
            if completed.returncode:
                raise RuntimeError(f"Benchmark worker failed: {completed.stderr[-2000:]}")
            report = read_json(destination.with_suffix(".json"))
            values = np.load(destination.with_suffix(".npy"), allow_pickle=False)
            if not np.isfinite(values).all():
                raise ValueError("Non-finite benchmark output")
            if reference is None:
                reference = values
            if values.shape != reference.shape:
                raise ValueError("Benchmark output shape mismatch")
            report["max_absolute_error"] = float(np.max(np.abs(values - reference)))
            report["parity_passed"] = bool(np.allclose(values, reference, rtol=1e-5, atol=1e-6))
            reports.append(report)
            print(
                {k: report[k] for k in ("batch", "mode", "warm_median_ms", "parity_passed")},
                flush=True,
            )
    result = {
        "scope": "encoder inference only; tokenization and model loading excluded",
        "input": "fixed tokenized benign request repeated within each batch; no traffic workload",
        "protocol": "fresh process for each mode/batch; first synchronized call then warm repetitions",
        "tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "results": reports,
    }
    write_json(output / "benchmark.json", result)
    if not all(r["parity_passed"] for r in reports):
        raise ValueError("Eager/JIT parity failed; timing results are not accepted")
    return result
