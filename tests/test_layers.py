#!/usr/bin/env python3
"""Unit tests for the building blocks in ``mamba_forest.layers``.

Run from the repository root:

    python -m pytest tests/test_layers.py
    python tests/test_layers.py          # same tests, without pytest
"""

from __future__ import annotations

import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mamba_forest.layers import (DynamicMultiModalFusion, Mamba2Block,  # noqa: E402
                                 StructureAwareStateFusion)


def test_mamba_block_preserves_shape():
  block = Mamba2Block(d_model=12, d_state=8)
  x = tf.random.normal((3, 7, 12))
  y = block(x, training=False)
  assert y.shape == x.shape, y.shape


def _perturbation(shape, scale: float = 5.0):
  """A perturbation that varies across channels.

  The block normalises over the channel axis before the recurrence, so a shift
  applied equally to every channel is removed by construction and would make
  any propagation test pass for the wrong reason.
  """
  return tf.random.normal(shape) * scale


def test_mamba_block_is_causal():
  """A time step must not be influenced by the steps that follow it."""
  block = Mamba2Block(d_model=8, d_state=4)
  x = tf.random.normal((1, 10, 8))
  block(x, training=False)  # build the weights

  # The separable convolution mixes one step to each side, so the strictly
  # causal part of the state update is checked two steps before the change.
  tail = x[:, 6:, :] + _perturbation((1, 4, 8))
  perturbed = tf.concat([x[:, :6, :], tail], axis=1)
  y_reference = block(x, training=False).numpy()
  y_perturbed = block(perturbed, training=False).numpy()

  max_diff = np.max(np.abs(y_reference[:, :5] - y_perturbed[:, :5]))
  assert max_diff < 1e-4, f"state leaked backwards, max difference {max_diff}"

  # The same perturbation must be visible at the steps that follow it, so the
  # test cannot pass by the block ignoring its input altogether.
  forward_diff = np.max(np.abs(y_reference[:, 6:] - y_perturbed[:, 6:]))
  assert forward_diff > 1e-5, "the perturbation does not reach its own step"


def test_mamba_block_state_carries_information():
  """The output at the last step must depend on the first step."""
  block = Mamba2Block(d_model=8, d_state=4)
  x = tf.random.normal((1, 6, 8))
  block(x, training=False)

  head = x[:, :1, :] + _perturbation((1, 1, 8))
  perturbed = tf.concat([head, x[:, 1:, :]], axis=1)
  y_reference = block(x, training=False).numpy()
  y_perturbed = block(perturbed, training=False).numpy()

  max_diff = np.max(np.abs(y_reference[:, -1] - y_perturbed[:, -1]))
  assert max_diff > 1e-6, "the recurrent state does not carry information"


def test_sasf_preserves_shape_and_locality():
  """SASF mixes direct neighbours only."""
  sasf = StructureAwareStateFusion()
  x = tf.random.normal((1, 6, 6, 5))
  y_reference = sasf(x).numpy()
  assert y_reference.shape == (1, 6, 6, 5), y_reference.shape

  perturbation = np.zeros((1, 6, 6, 5), dtype=np.float32)
  perturbation[0, 2, 2, :] = 10.0
  y_perturbed = sasf(x + tf.constant(perturbation)).numpy()

  neighbour = np.max(np.abs(y_reference[0, 2, 3] - y_perturbed[0, 2, 3]))
  distant = np.max(np.abs(y_reference[0, 5, 5] - y_perturbed[0, 5, 5]))
  assert neighbour > 1e-6, "a direct neighbour is not reached"
  assert distant < 1e-6, "a distant pixel must not be reached"


def test_fusion_output_shape_and_modality_count():
  fusion = DynamicMultiModalFusion(d_model=6, num_modalities=3)
  modalities = [tf.random.normal((2, 4, 4, 6)) for _ in range(3)]
  y = fusion(modalities, training=False)
  assert y.shape == (2, 4, 4, 6), y.shape

  try:
    fusion(modalities[:2], training=False)
  except ValueError:
    pass
  else:
    raise AssertionError("a wrong number of modalities must raise ValueError")


def test_fusion_uses_every_modality():
  """Every modality must be able to change the fused representation."""
  fusion = DynamicMultiModalFusion(d_model=6, num_modalities=3)
  modalities = [tf.random.normal((2, 4, 4, 6)) for _ in range(3)]
  reference = fusion(modalities, training=False).numpy()

  for index in range(3):
    changed = list(modalities)
    changed[index] = changed[index] * 3.0
    diff = np.max(np.abs(reference - fusion(changed, training=False).numpy()))
    assert diff > 1e-6, f"modality {index} does not influence the output"


if __name__ == "__main__":
  tests = [value for name, value in sorted(globals().items())
           if name.startswith("test_") and callable(value)]
  for test in tests:
    test()
    print(f"ok  {test.__name__}")
  print(f"\n{len(tests)} tests passed")
