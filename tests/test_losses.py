#!/usr/bin/env python3
"""Unit tests for the training objective in ``mamba_forest.losses``.

Run from the repository root:

    python -m pytest tests/test_losses.py
    python tests/test_losses.py          # same tests, without pytest
"""

from __future__ import annotations

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


if __name__ == "__main__":
  tests = [value for name, value in sorted(globals().items())
           if name.startswith("test_") and callable(value)]
  for test in tests:
    test()
    print(f"ok  {test.__name__}")
  print(f"\n{len(tests)} tests passed")
