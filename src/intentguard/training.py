"""Fixed-shape JIT training; last-batch padding carries zero sample weight."""

import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from .checkpoint import load_weights, save_weights
from .data import load_role
from .io import environment, read_json, sha256, write_json
from .model import IntentEncoder, ModelConfig, infer, parameter_count
from .tokenization import encode, fit_tokenizer, load_tokenizer
from .uncertainty import nll


def batches(encoded, labels, batch_size, order=None):
    order = np.arange(len(labels)) if order is None else np.asarray(order)
    for offset in range(0, len(order), batch_size):
        chosen = order[offset : offset + batch_size]
        real = len(chosen)
        ids = np.zeros((batch_size, encoded["input_ids"].shape[1]), dtype=np.int32)
        mask = np.zeros_like(ids)
        targets = np.zeros(batch_size, dtype=np.int32)
        weights = np.zeros(batch_size, dtype=np.float32)
        ids[:real] = encoded["input_ids"][chosen]
        mask[:real] = encoded["attention_mask"][chosen]
        targets[:real] = labels[chosen]
        weights[:real] = 1
        yield ids, mask, targets, weights


@nnx.jit
def train_step(model, optimizer, ids, mask, labels, weights, key):
    def objective(current):
        logits = current(ids, mask, key, training=True)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        return (losses * weights).sum() / jnp.maximum(weights.sum(), 1)

    loss, gradients = nnx.value_and_grad(objective)(model)
    norm = optax.tree.norm(gradients)
    optimizer.update(model, gradients)
    return loss, norm


def predict_logits(model, encoded, batch_size=64):
    result = []
    for ids, mask, _, weights in batches(
        encoded, np.zeros(len(encoded["input_ids"]), dtype=np.int32), batch_size
    ):
        output = np.asarray(infer(model, jnp.asarray(ids), jnp.asarray(mask)))
        result.append(output[: int(weights.sum())])
    return np.concatenate(result)


def train(
    data,
    output,
    seed=2026,
    epochs=20,
    batch_size=64,
    width=96,
    layers=2,
    max_length=32,
    vocab_size=4096,
    learning_rate=0.001,
    patience=5,
    tokenizer_from=None,
):
    if min(epochs, batch_size, width, layers, max_length, vocab_size, patience) < 1:
        raise ValueError("Training sizes must be positive")
    if learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("Learning rate must be finite and positive")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training output must be empty; existing evidence is never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    train_rows, y_train, labels = load_role(data, "train")
    select_rows, y_select, _ = load_role(data, "selection")
    if tokenizer_from is None:
        tokenizer = fit_tokenizer((r["text"] for r in train_rows), output / "tokenizer", vocab_size)
    else:
        prior = read_json(Path(tokenizer_from) / "protocol.json")
        if prior["manifest_sha256"] != sha256(Path(data) / "manifest.json"):
            raise ValueError("Tokenizer source belongs to a different dataset manifest")
        tokenizer = load_tokenizer(Path(tokenizer_from) / "tokenizer")
        tokenizer.save_pretrained(str(output / "tokenizer"))
    train_encoded = encode(tokenizer, [r["text"] for r in train_rows], max_length)
    select_encoded = encode(tokenizer, [r["text"] for r in select_rows], max_length)
    config = ModelConfig(
        len(tokenizer), len(labels), max_length=max_length, width=width, layers=layers
    )
    model = IntentEncoder(config, seed)
    steps = math.ceil(len(y_train) / batch_size)
    schedule = optax.warmup_cosine_decay_schedule(
        learning_rate * 0.1,
        learning_rate,
        steps,
        steps * max(epochs, 2),
        end_value=learning_rate * 0.05,
    )
    optimizer = nnx.Optimizer(
        model,
        optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=0.01)),
        wrt=nnx.Param,
    )
    write_json(
        output / "protocol.json",
        {
            "seed": seed,
            "epochs_budget": epochs,
            "patience": patience,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "model": config.to_dict(),
            "parameter_count": parameter_count(model),
            "manifest_sha256": sha256(Path(data) / "manifest.json"),
            "checkpoint_selection": "minimum in-scope selection-role NLL",
            "tokenizer_fit": "training texts only; WordPiece from scratch",
            "tokenizer_reused": tokenizer_from is not None,
            "tokenizer_sha256": sha256(output / "tokenizer" / "tokenizer.json"),
            "test_access_during_training": False,
            "environment": environment(),
        },
    )
    write_json(output / "labels.json", labels)
    generator = np.random.default_rng(seed)
    history = []
    best, stalled, global_step = float("inf"), 0, 0
    started, cpu_started = time.perf_counter(), time.process_time()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        total, count, max_norm = 0.0, 0, 0.0
        for ids, mask, targets, weights in batches(
            train_encoded, y_train, batch_size, generator.permutation(len(y_train))
        ):
            loss, norm = train_step(
                model,
                optimizer,
                jnp.asarray(ids),
                jnp.asarray(mask),
                jnp.asarray(targets),
                jnp.asarray(weights),
                jax.random.fold_in(jax.random.key(seed), global_step),
            )
            loss, norm = float(loss), float(norm)
            if not math.isfinite(loss) or not math.isfinite(norm):
                raise FloatingPointError(f"Non-finite loss/gradient at step {global_step}")
            total += loss * weights.sum()
            count += int(weights.sum())
            max_norm = max(max_norm, norm)
            global_step += 1
        logits = predict_logits(model, select_encoded, batch_size)
        selection_nll = nll(logits, y_select)
        improved = selection_nll < best
        if improved:
            best, stalled = selection_nll, 0
            save_weights(model, output)
        else:
            stalled += 1
        record = {
            "epoch": epoch,
            "training_nll": float(total / count),
            "selection_nll": selection_nll,
            "selection_accuracy": float((logits.argmax(1) == y_select).mean()),
            "max_preclip_gradient_norm": max_norm,
            "seconds": time.perf_counter() - epoch_started,
            "saved": improved,
        }
        history.append(record)
        write_json(output / "history.json", history)
        print(record, flush=True)
        if stalled >= patience:
            break
    write_json(
        output / "training_timing.json",
        {
            "wall_seconds": time.perf_counter() - started,
            "process_cpu_seconds": time.process_time() - cpu_started,
            "scope": "epoch loops, selection evaluation and checkpoint writes; includes first JIT compilation",
            "selected_epoch": min(history, key=lambda r: r["selection_nll"])["epoch"],
        },
    )
    # Load through the same strict path serving uses; do not return an unchecked final epoch.
    restored = load_weights(output)
    restored_logits = predict_logits(restored, select_encoded, batch_size)
    if not np.isclose(nll(restored_logits, y_select), best, rtol=1e-5, atol=1e-6):
        raise ValueError("Restored checkpoint does not reproduce selected NLL")
    return history
