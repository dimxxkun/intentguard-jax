"""Bounded inference shared by the CLI and HTTP adapter."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .checkpoint import load_weights
from .evaluation import verify_bundle
from .io import read_json, sha256
from .model import infer
from .tokenization import encode, load_tokenizer
from .uncertainty import prediction_sets, probabilities, route_mask


class Predictor:
    max_batch = 32
    max_characters = 500

    def __init__(self, bundle):
        bundle = Path(bundle)
        verify_bundle(bundle)
        self.model = load_weights(bundle)
        self.tokenizer = load_tokenizer(bundle / "tokenizer")
        self.labels = read_json(bundle / "labels.json")
        self.rule = read_json(bundle / "calibration.json")
        self.bundle_id = sha256(bundle / "bundle.json")
        if len(self.labels) != self.model.config.classes or len(set(self.labels)) != len(
            self.labels
        ):
            raise ValueError("Class order does not match the model")
        if not all(isinstance(label, str) and label for label in self.labels):
            raise ValueError("Invalid class names")
        if len(self.tokenizer) != self.model.config.vocab_size:
            raise ValueError("Tokenizer vocabulary does not match the model")

    def predict(self, texts, top_k=3):
        if not isinstance(texts, list) or not 1 <= len(texts) <= self.max_batch:
            raise ValueError("Provide between 1 and 32 texts")
        if type(top_k) is not int or not 1 <= top_k <= 5:
            raise ValueError("top_k must be between 1 and 5")
        if any(
            not isinstance(t, str) or not t.strip() or len(t) > self.max_characters for t in texts
        ):
            raise ValueError("Each text must contain 1 to 500 characters and cannot be whitespace")
        # Bounded shape buckets avoid a new JIT compilation for every request length.
        size = next(n for n in (1, 8, 32) if n >= len(texts))
        padded = texts + [""] * (size - len(texts))
        encoded = encode(self.tokenizer, padded, self.model.config.max_length)
        logits = np.asarray(
            infer(
                self.model,
                jnp.asarray(encoded["input_ids"], dtype=jnp.int32),
                jnp.asarray(encoded["attention_mask"], dtype=jnp.int32),
            )
        )[: len(texts)]
        probs = probabilities(logits, self.rule["temperature"])
        sets = prediction_sets(probs, self.rule["conformal_quantile"])
        accepted = route_mask(probs, self.rule["conformal_quantile"], self.rule["policy"])
        # Full token lengths are computed separately to make truncation visible to callers.
        full = self.tokenizer(texts, truncation=False, return_token_type_ids=False)["input_ids"]
        results = []
        for i, probability in enumerate(probs):
            order = np.argsort(-probability, kind="stable")[:top_k]
            count = int(sets[i].sum())
            reason = (
                "accepted"
                if accepted[i]
                else (
                    "empty_prediction_set"
                    if count == 0
                    else "ambiguous_prediction_set"
                    if count > 1
                    else "below_routing_threshold"
                )
            )
            results.append(
                {
                    "intent": self.labels[int(order[0])] if accepted[i] else None,
                    "abstained": not bool(accepted[i]),
                    "reason": reason,
                    "confidence": float(probability[order[0]]),
                    "top_k": [
                        {"intent": self.labels[int(j)], "probability": float(probability[j])}
                        for j in order
                    ],
                    "prediction_set": [self.labels[int(j)] for j in np.flatnonzero(sets[i])],
                    "truncated": len(full[i]) > self.model.config.max_length,
                    "original_token_count": len(full[i]),
                }
            )
        return {"bundle_id": self.bundle_id, "results": results}

    def warmup(self):
        for count in (1, 8, 32):
            self.predict(["warmup request"] * count)
