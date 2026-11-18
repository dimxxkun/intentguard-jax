"""Reproducible public commands; imports are deferred until CPU configuration is applied."""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="intentguard", description="Flax Transformer intent routing"
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=8,
        help="Bound CPU affinity where supported (default: 8 logical CPUs)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Download, verify, audit and partition CLINC150")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--source", help="Optional local copy of the pinned source JSON")
    prepare.add_argument("--seed", type=int, default=2026)
    train = sub.add_parser("train", help="Fit a train-only tokenizer and Flax encoder")
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True)
    for name, default in (
        ("seed", 2026),
        ("epochs", 20),
        ("batch-size", 64),
        ("width", 96),
        ("layers", 2),
        ("max-length", 32),
        ("vocab-size", 4096),
        ("patience", 5),
    ):
        train.add_argument(f"--{name}", type=int, default=default)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument(
        "--tokenizer-from", help="Reuse a tokenizer from a manifest-matched training run"
    )
    calibrate = sub.add_parser(
        "calibrate", help="Fit disjoint calibration roles and seal the bundle"
    )
    calibrate.add_argument("--data", required=True)
    calibrate.add_argument("--bundle", required=True)
    calibrate.add_argument("--alpha", type=float, default=0.1)
    calibrate.add_argument("--max-oos-accept", type=float, default=0.05)
    evaluate = sub.add_parser("evaluate", help="Evaluate the frozen bundle on the test role")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--bundle", required=True)
    evaluate.add_argument("--output", required=True)
    baseline = sub.add_parser("baseline", help="Train and evaluate TF-IDF/logistic regression")
    baseline.add_argument("--data", required=True)
    baseline.add_argument("--output", required=True)
    benchmark = sub.add_parser("benchmark", help="Compare eager/JIT inference in fresh processes")
    benchmark.add_argument("--bundle", required=True)
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--repeats", type=int, default=20)
    predict = sub.add_parser("predict", help="Predict locally; uncertain inputs explicitly abstain")
    predict.add_argument("--bundle", required=True)
    predict.add_argument("--text", action="append", required=True)
    predict.add_argument("--top-k", type=int, default=3)
    verify = sub.add_parser("verify", help="Verify bundle integrity and load it through inference")
    verify.add_argument("--bundle", required=True)
    serve = sub.add_parser("serve", help="Run the local FastAPI application")
    serve.add_argument("--bundle", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = vars(parser.parse_args())
    command, cpu_threads = args.pop("command"), args.pop("cpu_threads")
    if cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    import psutil

    process = psutil.Process()
    if hasattr(process, "cpu_affinity"):
        process.cpu_affinity(process.cpu_affinity()[:cpu_threads])
    try:
        if command == "prepare":
            from .data import prepare

            result = prepare(**args)
        elif command == "train":
            from .training import train

            history = train(**args)
            result = {"epochs_completed": len(history), "output": args["output"]}
        elif command in ("calibrate", "evaluate", "baseline"):
            from .evaluation import baseline, calibrate, evaluate_bundle

            result = {"calibrate": calibrate, "evaluate": evaluate_bundle, "baseline": baseline}[
                command
            ](**args)
            if command != "calibrate":
                result = {key: result[key] for key in ("model", "calibrated", "routing")}
                result["calibrated"].pop("reliability", None)
        elif command == "benchmark":
            from .benchmark import benchmark

            result = benchmark(**args, cpu_threads=cpu_threads)
            result = {
                "report": args["output"],
                "all_passed": all(r["parity_passed"] for r in result["results"]),
            }
        elif command in ("predict", "verify"):
            from .predictor import Predictor

            engine = Predictor(args["bundle"])
            result = (
                engine.predict(args["text"], args["top_k"])
                if command == "predict"
                else {
                    "verified": True,
                    "bundle_id": engine.bundle_id,
                    "classes": len(engine.labels),
                }
            )
        else:
            import uvicorn

            from .api import create_app

            uvicorn.run(
                create_app(args["bundle"]),
                host=args["host"],
                port=args["port"],
                workers=1,
                access_log=False,
            )
            return
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, FloatingPointError, RuntimeError) as error:
        print(f"intentguard: {error}", file=sys.stderr)
        raise SystemExit(1) from error
