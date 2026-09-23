#!/usr/bin/env python3
"""Unit tests for the segmentation network in ``mamba_forest.model``.

Run from the repository root:

    python -m pytest tests/test_model.py
    python tests/test_model.py           # same tests, without pytest
"""

from __future__ import annotations

import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mamba_forest.model import MambaForestSegmenter  # noqa: E402

NUM_CLASSES = 9


def _inputs(batch: int = 2, time: int = 4, size: int = 32):
  """A small multi-modal batch with the channel counts of ForTy v1."""
  return {
      "s2": tf.random.normal((batch, time, size, size, 10)),
      "elevation": tf.random.normal((batch, size, size, 3)),
      "climate": tf.random.normal((batch, time, 14)),
  }


def _model():
  return MambaForestSegmenter(
      num_classes=NUM_CLASSES, d_model=8, dropout_rate=0.1,
      d_state_temporal=4, d_state_spatial=4, bottleneck_channels=32)


def test_output_shape_matches_input_resolution():
  model, inputs = _model(), _inputs(size=32)
  logits = model(inputs, training=False)
  assert logits.shape == (2, 32, 32, NUM_CLASSES), logits.shape


def test_output_is_a_probability_distribution():
  model, inputs = _model(), _inputs(size=32)
  probabilities = model(inputs, training=False).numpy()
  assert np.all(probabilities >= 0.0)
  totals = probabilities.sum(axis=-1)
  assert np.allclose(totals, 1.0, atol=1e-4), totals.min()


def test_every_modality_influences_the_prediction():
  model, inputs = _model(), _inputs(size=32)
  reference = model(inputs, training=False).numpy()

  for key in ("s2", "elevation", "climate"):
    changed = dict(inputs)
    changed[key] = changed[key] * 3.0 + 1.0
    diff = np.max(np.abs(reference - model(changed, training=False).numpy()))
    assert diff > 1e-6, f"modality {key!r} does not influence the output"


def test_temporal_axis_is_read_per_pixel():
  """Reversing the order of the seasons must change the prediction.

  The optical tile here is constant over space, so the only structure left is
  temporal: the test fails if the phenological order is not actually read.
  """
  model = _model()
  batch, time, size = 1, 4, 32
  # A tile that is constant over space, so the only structure is temporal.
  series = tf.random.normal((batch, time, 1, 1, 10))
  optical = tf.tile(series, [1, 1, size, size, 1])
  inputs = {
      "s2": optical,
      "elevation": tf.random.normal((batch, size, size, 3)),
      "climate": tf.random.normal((batch, time, 14)),
  }
  reference = model(inputs, training=False).numpy()

  reversed_inputs = dict(inputs, s2=tf.reverse(optical, axis=[1]))
  reversed_output = model(reversed_inputs, training=False).numpy()

  diff = np.max(np.abs(reference - reversed_output))
  assert diff > 1e-6, "the temporal order of the seasons is ignored"


def test_gradients_are_finite():
  model, inputs = _model(), _inputs(size=32)
  labels = tf.one_hot(
      tf.random.uniform((2, 32, 32), maxval=NUM_CLASSES, dtype=tf.int32),
      NUM_CLASSES)

  with tf.GradientTape() as tape:
    predictions = model(inputs, training=True)
    loss = -tf.reduce_mean(
        tf.reduce_sum(labels * tf.math.log(predictions + 1e-7), axis=-1))
  gradients = tape.gradient(loss, model.trainable_variables)

  assert np.isfinite(loss.numpy()), loss.numpy()
  assert any(g is not None for g in gradients)
  for gradient, variable in zip(gradients, model.trainable_variables):
    if gradient is None:
      continue
    assert np.all(np.isfinite(gradient.numpy())), variable.name


if __name__ == "__main__":
  tests = [value for name, value in sorted(globals().items())
           if name.startswith("test_") and callable(value)]
  for test in tests:
    test()
    print(f"ok  {test.__name__}")
  print(f"\n{len(tests)} tests passed")
