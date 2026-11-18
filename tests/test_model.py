import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from intentguard.checkpoint import load_weights, save_weights
from intentguard.model import IntentEncoder, ModelConfig, infer
from intentguard.training import train_step


def small_model():
    return IntentEncoder(ModelConfig(32, 2, max_length=8, width=16, heads=4, layers=1, dropout=0))


def test_padding_tokens_cannot_change_valid_predictions_and_jit_agrees():
    model = small_model()
    ids = jnp.array([[2, 7, 3, 0, 0]], dtype=jnp.int32)
    mask = jnp.array([[1, 1, 1, 0, 0]], dtype=jnp.int32)
    changed = ids.at[:, 3:].set(17)
    expected = model(ids, mask)
    np.testing.assert_allclose(infer(model, changed, mask), expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        infer(model, ids[:, :3], mask[:, :3]), expected, rtol=1e-5, atol=1e-6
    )


def test_zero_weight_batch_padding_does_not_change_update():
    a, b = small_model(), small_model()
    oa = nnx.Optimizer(a, optax.sgd(0.01), wrt=nnx.Param)
    ob = nnx.Optimizer(b, optax.sgd(0.01), wrt=nnx.Param)
    ids = jnp.array([[2, 6, 3]], dtype=jnp.int32)
    mask = jnp.ones_like(ids)
    one, _ = train_step(a, oa, ids, mask, jnp.array([0]), jnp.ones(1), jax.random.key(0))
    padded, _ = train_step(
        b,
        ob,
        jnp.pad(ids, ((0, 3), (0, 0))),
        jnp.pad(mask, ((0, 3), (0, 0))),
        jnp.zeros(4, dtype=jnp.int32),
        jnp.array([1.0, 0.0, 0.0, 0.0]),
        jax.random.key(0),
    )
    assert float(one) == pytest.approx(float(padded), abs=1e-6)
    np.testing.assert_allclose(a(ids, mask), b(ids, mask), rtol=1e-5, atol=1e-6)


def test_checkpoint_roundtrip_and_corruption_rejection(tmp_path):
    model = small_model()
    save_weights(model, tmp_path)
    restored = load_weights(tmp_path)
    ids = jnp.array([[2, 5, 3]], dtype=jnp.int32)
    np.testing.assert_allclose(
        model(ids, jnp.ones_like(ids)), restored(ids, jnp.ones_like(ids)), rtol=1e-6, atol=1e-7
    )
    with (tmp_path / "weights.safetensors").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_weights(tmp_path)


def test_invalid_architecture_rejected():
    with pytest.raises(ValueError):
        ModelConfig(32, 2, width=15, heads=4)
    with pytest.raises(ValueError):
        ModelConfig(32, 2, layers=True)
