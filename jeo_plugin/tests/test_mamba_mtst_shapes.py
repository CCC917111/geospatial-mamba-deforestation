#!/usr/bin/env python3
"""Shape, gradient and causality tests for the Mamba-MTST model.

Run from the repository root (needs jax, flax and einops, no JEO checkout):

    python jeo_plugin/tests/test_mamba_mtst_shapes.py
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mamba_mtst  # noqa: E402  (path set up above)


def test_mamba_block_shapes():
  block = mamba_mtst.MambaBlock(d_model=32, d_state=8)
  x = jnp.ones((2, 10, 32))
  variables = block.init(jax.random.PRNGKey(0), x)
  y = block.apply(variables, x)
  assert y.shape == x.shape, y.shape


def test_mamba_block_is_causal():
  """Changing a late time step must not change earlier outputs."""
  block = mamba_mtst.MambaBlock(d_model=16, d_state=8)
  rng = jax.random.PRNGKey(0)
  x = jax.random.normal(rng, (1, 12, 16))
  variables = block.init(rng, x)

  y_ref = block.apply(variables, x)
  x_perturbed = x.at[:, 8:, :].set(x[:, 8:, :] + 10.0)
  y_perturbed = block.apply(variables, x_perturbed)

  max_diff = jnp.max(jnp.abs(y_ref[:, :8] - y_perturbed[:, :8]))
  assert max_diff < 1e-5, f"block is not causal, max difference {max_diff}"


def test_encoder_shapes():
  encoder = mamba_mtst.MambaEncoder(depth=2, d_model=32, d_state=8)
  x = jnp.ones((2, 10, 32))
  variables = encoder.init(jax.random.PRNGKey(0), x)
  assert encoder.apply(variables, x).shape == x.shape


def _multimodal_inputs(batch=2):
  return {
      "optical": jnp.ones((batch, 4, 32, 32, 10)),
      "climate": jnp.ones((batch, 4, 14)),          # no spatial extent
      "elevation": jnp.ones((batch, 32, 32, 3)),    # no time axis
  }


def _model(head):
  return mamba_mtst.Model(
      mods=["optical", "climate", "elevation"],
      patch_sizes={"optical": (2, 8, 8), "climate": (1, 1, 1),
                   "elevation": (1, 8, 8)},
      num_classes=9, d_model=32, d_state=8,
      temporal_depth=1, spatial_depth=1, head=head)


def test_classification_head():
  model, inputs = _model("classification"), _multimodal_inputs()
  variables = model.init(jax.random.PRNGKey(0), inputs, train=False)
  logits, out = model.apply(variables, inputs, train=False)
  assert logits.shape == (2, 9), logits.shape
  assert set(out) == {"tokens", "temporal_features", "spatial_features", "logits"}


def test_segmentation_head():
  model, inputs = _model("segmentation"), _multimodal_inputs()
  variables = model.init(jax.random.PRNGKey(0), inputs, train=False)
  logits, _ = model.apply(variables, inputs, train=False)
  # 4 x 4 patches of 8 x 8 pixels -> 32 x 32 class maps.
  assert logits.shape == (2, 32, 32, 9), logits.shape


def test_global_modality_first_is_not_dropped():
  """A global modality listed first must still reach the fused tokens."""
  inputs = _multimodal_inputs()
  model = mamba_mtst.Model(
      mods=["climate", "optical"],
      patch_sizes={"climate": (1, 1, 1), "optical": (2, 8, 8)},
      num_classes=9, d_model=32, d_state=8,
      temporal_depth=1, spatial_depth=1, head="classification")
  variables = model.init(jax.random.PRNGKey(0), inputs, train=False)

  logits_a, _ = model.apply(variables, inputs, train=False)
  perturbed = dict(inputs, climate=inputs["climate"] * 3.0)
  logits_b, _ = model.apply(variables, perturbed, train=False)
  assert jnp.max(jnp.abs(logits_a - logits_b)) > 1e-6, (
      "the climate modality does not influence the output")


def test_gradients_are_finite():
  model, inputs = _model("classification"), _multimodal_inputs()
  rng, dropout_rng = jax.random.split(jax.random.PRNGKey(0))
  variables = model.init(rng, inputs, train=False)

  def loss_fn(params):
    logits, _ = model.apply({"params": params}, inputs, train=True,
                            rngs={"dropout": dropout_rng})
    labels = jax.nn.one_hot(jnp.array([0, 1]), 9)
    return -jnp.mean(jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1))

  loss, grads = jax.value_and_grad(loss_fn)(variables["params"])
  assert jnp.isfinite(loss), loss
  assert all(jnp.all(jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads))


if __name__ == "__main__":
  tests = [value for name, value in sorted(globals().items())
           if name.startswith("test_") and callable(value)]
  for test in tests:
    test()
    print(f"ok  {test.__name__}")
  print(f"\n{len(tests)} tests passed")
