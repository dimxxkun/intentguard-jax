"""Regenerate committed plots and summaries from retained measurements, never invented values."""

import argparse
import shutil
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from intentguard.io import read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root)
    results = root / "results"
    previews = root / "runs" / "report-preview"
    previews.mkdir(parents=True, exist_ok=True)
    colors = ["#147d92", "#6c92ac", "#d18a35"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#fafbfc",
            "axes.facecolor": "#fafbfc",
            "svg.fonttype": "none",
            "svg.hashsalt": "intentguard-jax",
        }
    )
    names = ["seed-2026", "seed-2027", "baseline"]
    labels = ["Transformer 2026", "Transformer 2027", "TF–IDF baseline"]
    reports = [read_json(results / name / "evaluation.json") for name in names]
    for seed in (2026, 2027):
        source = root / "runs" / f"seed-{seed}"
        target = results / f"seed-{seed}"
        for name in (
            "protocol.json",
            "history.json",
            "training_timing.json",
            "calibration.json",
            "model.json",
        ):
            shutil.copyfile(source / name, target / name)
    for name in ("manifest.json", "audit.json"):
        shutil.copyfile(root / "data" / "clinc150" / name, results / name)
    hashes = {
        str(seed): sha256(root / "runs" / f"seed-{seed}" / "tokenizer" / "tokenizer.json")
        for seed in (2026, 2027)
    }
    if len(set(hashes.values())) != 1:
        raise ValueError("Training runs did not share the same tokenizer")
    summary = {
        "primary_seed": 2026,
        "tokenizer_sha256": hashes,
        "classification_accuracy_mean": statistics.mean(
            r["calibrated"]["accuracy"] for r in reports[:2]
        ),
        "classification_accuracy_sample_std": statistics.stdev(
            r["calibrated"]["accuracy"] for r in reports[:2]
        ),
        "interpretation": "Two seeds on one cleaned split; not a confidence interval or a shift evaluation",
        "models": [
            {
                "name": name,
                "accuracy": r["calibrated"]["accuracy"],
                "macro_f1": r["calibrated"]["macro_f1"],
                "ece": r["calibrated"]["ece"],
                "conformal_coverage": r["sets"]["in_scope_coverage"],
                **r["routing"],
            }
            for name, r in zip(names, reports)
        ],
    }
    write_json(results / "summary.json", summary)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), layout="constrained")
    specs = [
        ("All in-scope accuracy", [r["calibrated"]["accuracy"] for r in reports]),
        ("In-scope routing coverage", [r["routing"]["in_scope_coverage"] for r in reports]),
        (
            "Out-of-scope false acceptance",
            [r["routing"]["out_of_scope_false_accept_rate"] for r in reports],
        ),
    ]
    for ax, (title, values) in zip(axes, specs):
        bars = ax.barh(labels, [v * 100 for v in values], color=colors)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", weight="bold", pad=14)
        ax.set_xlabel("Percent of the stated population")
        ax.set_xlim(0, 105 if "false" not in title else 10)
        ax.bar_label(bars, labels=[f"{v:.2%}" for v in values], padding=4)
    fig.suptitle(
        "IntentGuard • held-out test results", x=0.01, ha="left", fontsize=16, weight="bold"
    )
    fig.savefig(results / "comparison.svg", metadata={"Date": None})
    fig.savefig(previews / "comparison.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for seed, color in zip((2026, 2027), colors):
        history = read_json(results / f"seed-{seed}" / "history.json")
        for field, style in (("training_nll", "-"), ("selection_nll", "--")):
            axes[0].plot(
                [r["epoch"] for r in history],
                [r[field] for r in history],
                style,
                color=color,
                label=f"{seed} {field.replace('_', ' ')}",
            )
    axes[0].set(title="Checkpoint selection reveals overfitting", xlabel="Epoch", ylabel="NLL")
    axes[0].legend(fontsize=8)
    for field, color in (("uncalibrated", "#d18a35"), ("calibrated", "#147d92")):
        bins = [b for b in reports[0][field]["reliability"] if b["count"]]
        axes[1].plot(
            [b["confidence"] for b in bins],
            [b["accuracy"] for b in bins],
            "o-",
            color=color,
            label=field,
        )
    axes[1].plot([0, 1], [0, 1], "--", color="#9ba8b4")
    axes[1].set(
        title="Primary seed: confidence versus accuracy",
        xlabel="Mean confidence in bin",
        ylabel="Accuracy in bin",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axes[1].legend()
    fig.savefig(results / "calibration-and-training.svg", metadata={"Date": None})
    fig.savefig(previews / "calibration-and-training.png", dpi=130)
    plt.close(fig)
    benchmark = root / "runs" / "benchmark" / "benchmark.json"
    if benchmark.exists():
        report = read_json(benchmark)
        write_json(results / "benchmark.json", report)
        fig, ax = plt.subplots(figsize=(8.5, 4), layout="constrained")
        for i, mode in enumerate(("eager", "jit")):
            rows = [r for r in report["results"] if r["mode"] == mode]
            xs = [j + (i - 0.5) * 0.36 for j in range(3)]
            bars = ax.bar(
                xs,
                [r["warm_median_ms"] for r in rows],
                width=0.36,
                label=mode.upper(),
                color=colors[i],
            )
            ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set_xticks([0, 1, 2], ["Batch 1", "Batch 8", "Batch 32"])
        ax.set(
            ylabel="Warm median encoder latency (ms)",
            title="JAX eager versus JIT • synchronized CPU calls",
        )
        ax.legend()
        fig.savefig(results / "latency.svg", metadata={"Date": None})
        fig.savefig(previews / "latency.png", dpi=130)
        plt.close(fig)
    # Matplotlib's SVG path formatter leaves spaces before line breaks.
    for path in results.glob("*.svg"):
        path.write_text(
            "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print("Reports and figures generated from retained experiment files.")


if __name__ == "__main__":
    main()
