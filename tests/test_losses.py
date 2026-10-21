#!/usr/bin/env python3
"""Unit tests for the training objective in ``mamba_forest.losses``.

Run from the repository root:

    python -m pytest tests/test_losses.py
    python tests/test_losses.py          # same tests, without pytest
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mamba_forest.losses import DEFAULT_CLASS_WEIGHTS, ComboLoss  # noqa: E402

NUM_CLASSES = 9


def _one_hot(indices):
  return tf.one_hot(tf.constant(indices, dtype=tf.int32), NUM_CLASSES)


def _labels_and_prediction(label_index: int, predicted_index: int,
                           confidence: float = 0.9):
  """A 1x2x2 tile with a single true class and a single predicted class."""
  labels = _one_hot(np.full((1, 2, 2), label_index))
  wrong = (1.0 - confidence) / (NUM_CLASSES - 1)
  prediction = np.full((1, 2, 2, NUM_CLASSES), wrong, dtype=np.float32)
  prediction[..., predicted_index] = confidence
  return labels, tf.constant(prediction)


def test_correct_prediction_scores_better_than_wrong_one():
  loss = ComboLoss(NUM_CLASSES)
  correct = loss(*_labels_and_prediction(1, 1))
  wrong = loss(*_labels_and_prediction(1, 4))
  assert float(correct) < float(wrong), (float(correct), float(wrong))


def test_class_weights_penalise_the_rare_classes_more():
  """Missing planted forest must cost more than missing an unweighted class."""
  loss = ComboLoss(NUM_CLASSES, alpha=1.0)  # cross-entropy only
  planted = float(loss(*_labels_and_prediction(2, 0)))
  other = float(loss(*_labels_and_prediction(4, 0)))
  assert planted > other, (planted, other)
  assert np.isclose(planted / other, DEFAULT_CLASS_WEIGHTS[2], rtol=1e-3)


def test_uniform_weights_are_class_independent():
  loss = ComboLoss(NUM_CLASSES, alpha=1.0, class_weights=None)
  planted = float(loss(*_labels_and_prediction(2, 0)))
  other = float(loss(*_labels_and_prediction(4, 0)))
  assert np.isclose(planted, other, rtol=1e-5), (planted, other)


def test_absent_classes_do_not_inflate_the_dice_term():
  """A class in neither the labels nor the prediction must not score."""
  loss = ComboLoss(NUM_CLASSES, alpha=0.0)  # Dice only
  labels, prediction = _labels_and_prediction(1, 1, confidence=1.0)
  # One class present and perfectly predicted: the Dice loss is ~0 only if the
  # eight absent classes are excluded from the average.
  assert float(loss(labels, prediction)) < 1e-3, float(loss(labels, prediction))


def test_alpha_and_weight_validation():
  for bad_alpha in (-0.1, 1.5):
    try:
      ComboLoss(NUM_CLASSES, alpha=bad_alpha)
    except ValueError:
      continue
    raise AssertionError(f"alpha={bad_alpha} must be rejected")

  try:
    ComboLoss(NUM_CLASSES, class_weights=(1.0, 2.0))
  except ValueError:
    return
  raise AssertionError("a wrong number of class weights must be rejected")


def test_numpy_list_and_tuple_weights_preserve_loss_gradients_and_config():
  labels, prediction = _labels_and_prediction(2, 1, confidence=0.8)
  expected_loss, expected_gradients = None, None
  for weights in (DEFAULT_CLASS_WEIGHTS, list(DEFAULT_CLASS_WEIGHTS),
                  np.asarray(DEFAULT_CLASS_WEIGHTS, dtype=np.float64)):
    loss = ComboLoss(np.int64(NUM_CLASSES), class_weights=weights)
    assert loss.class_weights == DEFAULT_CLASS_WEIGHTS
    assert all(type(weight) is float for weight in loss.class_weights)
    restored = ComboLoss.from_config(json.loads(json.dumps(loss.get_config())))
    assert restored.get_config() == loss.get_config()
    logits = tf.Variable(tf.math.log(prediction))
    with tf.GradientTape() as tape:
      value = loss(labels, tf.nn.softmax(logits))
    gradients = tape.gradient(value, logits).numpy()
    assert np.isfinite(float(value))
    assert np.isfinite(gradients).all()
    assert np.any(gradients != 0)
    assert np.isclose(float(restored(labels, prediction)),
                      float(loss(labels, prediction)))
    if expected_loss is None:
      expected_loss, expected_gradients = float(value), gradients
    else:
      assert float(value) == expected_loss
      np.testing.assert_array_equal(gradients, expected_gradients)


def test_zero_weight_array_is_preserved_and_none_remains_uniform():
  labels, prediction = _labels_and_prediction(2, 0)
  zero_weighted = ComboLoss(
      NUM_CLASSES, alpha=1.0, class_weights=np.zeros(NUM_CLASSES))
  assert zero_weighted.class_weights == (0.0,) * NUM_CLASSES
  assert float(zero_weighted(labels, prediction)) == 0.0
  uniform = ComboLoss(NUM_CLASSES, alpha=1.0, class_weights=None)
  assert uniform.class_weights is None
  assert float(uniform(labels, prediction)) > 0.0
  restored = ComboLoss.from_config(uniform.get_config())
  assert restored.class_weights is None


def test_invalid_loss_configuration_is_rejected_before_tensor_math():
  invalid = [
      {"num_classes": 0}, {"num_classes": -1}, {"num_classes": 9.0},
      {"num_classes": True}, {"num_classes": "9"},
      {"alpha": float("nan")}, {"alpha": float("inf")},
      {"label_smoothing": -0.1}, {"label_smoothing": 1.1},
      {"label_smoothing": float("nan")}, {"label_smoothing": float("inf")},
      {"smooth": 0.0}, {"smooth": -1e-6},
      {"smooth": float("nan")}, {"smooth": float("inf")},
      {"class_weights": np.ones(NUM_CLASSES - 1)},
      {"class_weights": np.full(NUM_CLASSES, -1.0)},
      {"class_weights": np.full(NUM_CLASSES, float("nan"))},
      {"class_weights": np.full(NUM_CLASSES, float("inf"))},
      {"class_weights": 1.0}, {"class_weights": ["invalid"] * NUM_CLASSES},
  ]
  for overrides in invalid:
    config = {"num_classes": NUM_CLASSES, **overrides}
    try:
      ComboLoss(**config)
    except ValueError:
      continue
    raise AssertionError(f"invalid loss configuration accepted: {overrides}")


def test_smoothing_boundaries_keep_loss_and_gradients_finite():
  labels, prediction = _labels_and_prediction(2, 0)
  for smoothing in (0.0, 1.0):
    loss = ComboLoss(NUM_CLASSES, label_smoothing=smoothing)
    logits = tf.Variable(tf.math.log(prediction))
    with tf.GradientTape() as tape:
      value = loss(labels, tf.nn.softmax(logits))
    assert np.isfinite(float(value))
    assert np.isfinite(tape.gradient(value, logits).numpy()).all()


if __name__ == "__main__":
  tests = [value for name, value in sorted(globals().items())
           if name.startswith("test_") and callable(value)]
  for test in tests:
    test()
    print(f"ok  {test.__name__}")
  print(f"\n{len(tests)} tests passed")
