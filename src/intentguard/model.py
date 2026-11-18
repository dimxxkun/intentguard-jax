"""Small pre-norm Transformer encoder, trained from scratch with explicit PRNG keys."""

from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
from flax import nnx


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    classes: int
    max_length: int = 32
    width: int = 96
    heads: int = 4
    layers: int = 2
    ff_multiplier: int = 2
    dropout: float = 0.1

    def __post_init__(self):
        sizes = (
            self.vocab_size,
            self.classes,
            self.max_length,
            self.width,
            self.heads,
            self.layers,
            self.ff_multiplier,
        )
        if any(type(v) is not int or v < 1 for v in sizes):
            raise ValueError("Model dimensions must be positive integers")
        if self.width % self.heads or not 0 <= self.dropout < 1:
            raise ValueError("Heads must divide width; dropout must be in [0,1)")

    def to_dict(self):
        return asdict(self)


def dropout(x, key, rate):
    keep_probability = 1.0 - rate
    keep = jax.random.bernoulli(key, keep_probability, x.shape)
    return jnp.where(keep, x / keep_probability, 0)


class EncoderBlock(nnx.Module):
    def __init__(self, config, rngs):
        self.width, self.heads, self.rate = config.width, config.heads, config.dropout
        self.norm1 = nnx.LayerNorm(config.width, rngs=rngs)
        self.norm2 = nnx.LayerNorm(config.width, rngs=rngs)
        self.qkv = nnx.Linear(config.width, 3 * config.width, rngs=rngs)
        self.projection = nnx.Linear(config.width, config.width, rngs=rngs)
        self.ff1 = nnx.Linear(config.width, config.width * config.ff_multiplier, rngs=rngs)
        self.ff2 = nnx.Linear(config.width * config.ff_multiplier, config.width, rngs=rngs)

    def __call__(self, x, mask, key, training):
        batch, length, _ = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(
            batch, length, 3, self.heads, self.width // self.heads
        )
        q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(q.shape[-1])
        scores = jnp.where(mask[:, None, None, :].astype(bool), scores, -1e9)
        attention = jax.nn.softmax(scores, axis=-1)
        context = jnp.einsum("bhqk,bhkd->bhqd", attention, v)
        context = context.transpose(0, 2, 1, 3).reshape(batch, length, self.width)
        update = self.projection(context)
        keys = jax.random.split(key, 2)
        x = x + (dropout(update, keys[0], self.rate) if training else update)
        update = self.ff2(jax.nn.gelu(self.ff1(self.norm2(x))))
        return x + (dropout(update, keys[1], self.rate) if training else update)


class IntentEncoder(nnx.Module):
    def __init__(self, config: ModelConfig, seed=2026):
        self.config = config
        rngs = nnx.Rngs(seed)
        self.embedding = nnx.Embed(
            config.vocab_size, config.width, embedding_init=nnx.initializers.normal(0.02), rngs=rngs
        )
        self.position = nnx.Param(
            jax.random.normal(rngs.params(), (config.max_length, config.width)) * 0.02
        )
        self.blocks = nnx.List([EncoderBlock(config, rngs) for _ in range(config.layers)])
        self.norm = nnx.LayerNorm(config.width, rngs=rngs)
        self.head = nnx.Linear(config.width, config.classes, rngs=rngs)

    def __call__(self, ids, mask, key=None, training=False):
        key = jax.random.key(0) if key is None else key
        x = self.embedding(ids) + self.position[None, : ids.shape[1], :]
        for i, block in enumerate(self.blocks):
            x = block(x, mask, jax.random.fold_in(key, i), training)
        x = self.norm(x) * mask[..., None]
        pooled = x.sum(axis=1) / jnp.maximum(mask.sum(axis=1, keepdims=True), 1)
        return self.head(pooled)


@nnx.jit
def infer(model, ids, mask):
    return model(ids, mask)


def parameter_count(model):
    return sum(v.size for v in jax.tree.leaves(nnx.state(model, nnx.Param)))
