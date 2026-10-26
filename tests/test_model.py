#!/usr/bin/env python3
"""Unit tests for the segmentation network in ``mamba_forest.model``.

Run from the repository root:

    python -m pytest tests/test_model.py
    python tests/test_model.py           # same tests, without pytest
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mamba_forest.model import MambaForestSegmenter  # noqa: E402
from mamba_forest.layers import MambaBlock  # noqa: E402

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


def _variant_model(variant="cnn_only"):
  return MambaForestSegmenter(
      num_classes=NUM_CLASSES, d_model=4, dropout_rate=0.0,
      d_state_temporal=2, d_state_spatial=2, bottleneck_channels=8,
      model_variant=variant)


def test_cnn_only_has_no_ssm_layers_and_fewer_parameters():
  inputs = _inputs(batch=1, size=8)
  baseline = _variant_model()
  probabilities = baseline(inputs, training=False).numpy()
  assert probabilities.shape == (1, 8, 8, NUM_CLASSES)
  assert np.isfinite(probabilities).all() and (probabilities >= 0).all()
  np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0, atol=1e-6)
  assert not any(isinstance(layer, MambaBlock)
                 for layer in baseline._flatten_layers(recursive=True))
  full = _variant_model("cnn_ssm")
  full(inputs, training=False)
  assert sum(isinstance(layer, MambaBlock)
             for layer in full._flatten_layers(recursive=True)) == 2
  assert baseline.count_params() < full.count_params()


def test_cnn_only_temporal_mean_ignores_optical_season_order():
  model, inputs = _variant_model(), _inputs(batch=1, size=8)
  reference = model(inputs, training=False).numpy()
  reordered = dict(inputs, s2=tf.gather(inputs["s2"], [2, 0, 3, 1], axis=1))
  np.testing.assert_allclose(
      model(reordered, training=False).numpy(), reference, atol=1e-6)


def test_cnn_only_all_trainable_weights_have_finite_gradients():
  model, inputs = _variant_model(), _inputs(batch=1, size=8)
  labels = tf.one_hot(
      tf.random.uniform((1, 8, 8), maxval=NUM_CLASSES, dtype=tf.int32),
      NUM_CLASSES)
  with tf.GradientTape() as tape:
    predictions = model(inputs, training=True)
    loss = -tf.reduce_mean(
        tf.reduce_sum(labels * tf.math.log(predictions + 1e-7), axis=-1))
  gradients = tape.gradient(loss, model.trainable_variables)
  assert np.isfinite(loss.numpy())
  assert gradients
  for gradient, variable in zip(gradients, model.trainable_variables):
    assert gradient is not None, variable.name
    assert np.isfinite(gradient.numpy()).all(), variable.name


def test_cnn_only_config_and_checkpoint_roundtrip_preserve_predictions():
  model, inputs = _variant_model(), _inputs(batch=1, size=8)
  expected = model(inputs, training=False).numpy()
  assert model.get_config()["model_variant"] == "cnn_only"
  restored = MambaForestSegmenter.from_config(model.get_config())
  restored(inputs, training=False)
  with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "best.weights.h5"
    model.save_weights(path)
    restored.load_weights(path)
  assert restored.model_variant == "cnn_only"
  np.testing.assert_allclose(restored(inputs, training=False), expected, atol=1e-7)


def test_model_variant_defaults_to_ssm_and_rejects_unknown_names():
  assert _model().get_config()["model_variant"] == "cnn_ssm"
  for variant in ("unknown", "", None, True, 7):
    with pytest.raises(ValueError, match="model_variant"):
      _variant_model(variant)


def test_final_temporal_readout_restores_full_readout_checkpoint_predictions():
  """An old full-sequence temporal forward pass saves compatible weights."""
  previous = tf.keras.mixed_precision.global_policy()
  tf.keras.mixed_precision.set_global_policy("float32")
  try:
    tf.keras.utils.set_random_seed(304)
    reference = _variant_model("cnn_ssm")
    full_call = reference.temporal_mamba.call

    def legacy_temporal_call(x, training=False, return_last=False):
      full = full_call(x, training=training, return_last=False)
      return full[:, -1, :] if return_last else full

    reference.temporal_mamba.call = legacy_temporal_call
    inputs = _inputs(batch=1, size=8)
    expected = reference(inputs, training=False).numpy()
    restored = MambaForestSegmenter.from_config(reference.get_config())
    restored(inputs, training=False)
    assert restored.count_params() == reference.count_params()
    assert [tuple(weight.shape) for weight in restored.weights] == [
        tuple(weight.shape) for weight in reference.weights]
    with tempfile.TemporaryDirectory() as directory:
      checkpoint = Path(directory) / "legacy.weights.h5"
      reference.save_weights(checkpoint)
      restored.load_weights(checkpoint)
    np.testing.assert_allclose(restored(inputs, training=False).numpy(), expected,
                               rtol=2e-5, atol=2e-6)
    assert "return_last" not in restored.get_config()
  finally:
    tf.keras.mixed_precision.set_global_policy(previous)


if __name__ == "__main__":
  tests = [value for name, value in sorted(globals().items())
           if name.startswith("test_") and callable(value)]
  for test in tests:
    test()
    print(f"ok  {test.__name__}")
  print(f"\n{len(tests)} tests passed")
