import argparse
import threading
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["eager", "jit"], required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--cpu-threads", type=int, required=True)
    args = parser.parse_args()
    import psutil

    process = psutil.Process()
    if hasattr(process, "cpu_affinity"):
        process.cpu_affinity(process.cpu_affinity()[: args.cpu_threads])
    import jax
    import jax.numpy as jnp
    import numpy as np

    from .io import environment, write_json
    from .model import infer
    from .predictor import Predictor
    from .tokenization import encode

    predictor = Predictor(args.bundle)
    encoded = encode(
        predictor.tokenizer,
        ["please check my account balance"] * args.batch,
        predictor.model.config.max_length,
    )
    ids = jnp.asarray(encoded["input_ids"], dtype=jnp.int32)
    mask = jnp.asarray(encoded["attention_mask"], dtype=jnp.int32)
    call = (
        (lambda: infer(predictor.model, ids, mask))
        if args.mode == "jit"
        else (lambda: predictor.model(ids, mask))
    )
    baseline = process.memory_info().rss
    peak = [baseline]
    stopped = threading.Event()

    def sample():
        while not stopped.wait(0.005):
            peak[0] = max(peak[0], process.memory_info().rss)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    wall, cpu = [], []
    try:
        for _ in range(args.repeats + 1):
            before_cpu, before = time.process_time(), time.perf_counter()
            result = call()
            jax.block_until_ready(result)
            wall.append((time.perf_counter() - before) * 1000)
            cpu.append((time.process_time() - before_cpu) * 1000)
            peak[0] = max(peak[0], process.memory_info().rss)
    finally:
        stopped.set()
        sampler.join()
    destination = Path(args.output)
    np.save(destination.with_suffix(".npy"), np.asarray(result))
    report = {
        "mode": args.mode,
        "batch": args.batch,
        "first_call_ms": wall[0],
        "warm_ms": wall[1:],
        "warm_median_ms": float(np.median(wall[1:])),
        "warm_p95_ms": float(np.percentile(wall[1:], 95)),
        "first_process_cpu_ms": cpu[0],
        "warm_process_cpu_ms": cpu[1:],
        "baseline_rss_mib": baseline / 2**20,
        "sampled_peak_rss_mib": peak[0] / 2**20,
        "memory_sampling_ms": 5,
        "environment": environment(),
        "bundle_id": predictor.bundle_id,
        "first_call_scope": "after model/tokenizer loading; JIT compilation included in jit mode",
    }
    write_json(destination.with_suffix(".json"), report)


if __name__ == "__main__":
    main()
