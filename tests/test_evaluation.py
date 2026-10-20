"""Evaluation correctness and checkpoint provenance without remote data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mamba_forest import data as data_lib  # noqa: E402
from mamba_forest import evaluate as evaluation_cli  # noqa: E402
from mamba_forest.train import evaluate, forest_scores  # noqa: E402


class PredictionModel(tf.keras.Model):
  """Returns the supplied probabilities, with a checkpointable scalar weight."""

  def __init__(self, **kwargs):
    super().__init__()
    self.offset = self.add_weight(shape=(), initializer="zeros")

  def call(self, inputs, training=False):
    assert training is False
    return inputs["s2"] + self.offset


def _dataset(labels, predictions, batch_size=2, num_classes=4):
  labels = tf.convert_to_tensor(labels, dtype=tf.int32)
  predictions = tf.convert_to_tensor(predictions, dtype=tf.int32)
  count = tf.shape(labels)[0]
  examples = {
      "s2": tf.one_hot(predictions, num_classes),
      "elevation": tf.zeros((count, 2, 2, 3)),
      "climate": tf.zeros((count, 4, 14)),
      "labels": tf.one_hot(labels, num_classes),
  }
  return tf.data.Dataset.from_tensor_slices(examples).batch(batch_size)


def _examples():
  labels = [[[0, 0], [1, 1]], [[1, 2], [2, 2]], [[0, 1], [2, 0]]]
  predictions = [[[0, 1], [1, 2]], [[0, 2], [2, 0]], [[0, 1], [1, 0]]]
  return labels, predictions


def test_metrics_match_hand_computed_confusion_and_absent_class_policy():
  accuracy, mean_iou, f1, confusion = evaluate(
      PredictionModel(), _dataset(*_examples()), 4, return_confusion=True)
  expected = [[3, 1, 0, 0], [1, 2, 1, 0], [1, 1, 2, 0], [0, 0, 0, 0]]
  np.testing.assert_array_equal(confusion, expected)
  assert confusion.dtype == np.int64
  assert accuracy == pytest.approx(7 / 12)
  assert mean_iou == pytest.approx((3 / 6 + 2 / 6 + 2 / 5) / 3)
  np.testing.assert_allclose(f1, [6 / 9, 4 / 8, 4 / 7, 0])


def test_metrics_do_not_depend_on_batch_boundaries():
  model = PredictionModel()
  single = evaluate(model, _dataset(*_examples(), batch_size=1), 4)
  uneven = evaluate(model, _dataset(*_examples(), batch_size=2), 4)
  assert len(single) == 3
  np.testing.assert_allclose(single[:2], uneven[:2])
  np.testing.assert_array_equal(single[2], uneven[2])


def test_predicted_only_class_is_included_in_mean_iou():
  dataset = _dataset([[[0, 0], [0, 0]]], [[[0, 1], [0, 1]]])
  accuracy, mean_iou, f1 = evaluate(PredictionModel(), dataset, 4)
  assert accuracy == pytest.approx(0.5)
  assert mean_iou == pytest.approx(0.25)
  np.testing.assert_allclose(f1, [2 / 3, 0, 0, 0])


def test_empty_and_zero_pixel_datasets_are_rejected():
  dataset = _dataset(*_examples())
  with pytest.raises(ValueError, match="contains no pixels"):
    evaluate(PredictionModel(), dataset.take(0), 4)
  empty_pixels = _dataset(np.zeros((1, 0, 0), np.int32),
                          np.zeros((1, 0, 0), np.int32))
  with pytest.raises(ValueError, match="contains no pixels"):
    evaluate(PredictionModel(), empty_pixels, 4)


def test_forest_scores_use_declared_class_indices():
  f1 = np.arange(data_lib.NUM_CLASSES, dtype=np.float64) / 10
  scores = forest_scores(f1)
  assert scores["overall_f1"] == pytest.approx(np.mean(f1))
  assert scores["forests_f1"] == pytest.approx(0.2)
  assert scores["natural_forest_f1"] == pytest.approx(0.1)
  assert scores["planted_forest_f1"] == pytest.approx(0.2)
  assert scores["tree_crops_f1"] == pytest.approx(0.3)


def _cli_args(tmp_path):
  return argparse.Namespace(
      weights=str(tmp_path / "best.weights.h5"),
      gcs_root=str(tmp_path / "records"), test_shards=2, batch_size=2,
      d_model=8, dropout=0.0, d_state_temporal=2, d_state_spatial=2,
      bottleneck_channels=8, output_dir=str(tmp_path))


def test_cli_retains_recomputable_metrics_and_checkpoint_provenance(
      tmp_path, monkeypatch):
  args = _cli_args(tmp_path)
  dataset = _dataset(*_examples(), num_classes=data_lib.NUM_CLASSES)
  model = PredictionModel()
  model(data_lib.split_inputs_labels(next(iter(dataset)))[0])
  model.save_weights(args.weights)
  monkeypatch.setattr(evaluation_cli, "parse_args", lambda: args)
  monkeypatch.setattr(evaluation_cli, "MambaForestSegmenter", PredictionModel)
  monkeypatch.setattr(data_lib, "make_dataset", lambda *a, **kw: dataset)
  evaluation_cli.main()

  metrics = json.loads((tmp_path / "test_metrics.json").read_text())
  assert metrics["weights_sha256"] == hashlib.sha256(
      (tmp_path / "best.weights.h5").read_bytes()).hexdigest()
  assert metrics["gcs_root"] == args.gcs_root
  assert metrics["split"] == "test"
  assert metrics["shard_paths"] == data_lib.selected_shard_paths(
      "test", args.test_shards, gcs_root=args.gcs_root)
  assert metrics["num_pixels"] == 12
  confusion = np.asarray(metrics["confusion_matrix"])
  assert metrics["pixel_accuracy"] == pytest.approx(
      np.trace(confusion) / confusion.sum())
  assert metrics["class_support"] == dict(
      zip(data_lib.CLASS_NAMES, confusion.sum(axis=1)))
  assert metrics["confusion_axes"] == {"rows": "true", "columns": "predicted"}
  assert metrics["runtime"]["tensorflow_version"] == tf.__version__
  assert metrics["runtime"]["precision_policy"] == (
      tf.keras.mixed_precision.global_policy().name)
  assert metrics["model_config"]


def test_cli_rejects_empty_test_split_before_loading_weights(
      tmp_path, monkeypatch):
  args = _cli_args(tmp_path)
  dataset = _dataset(*_examples()).take(0)
  monkeypatch.setattr(evaluation_cli, "parse_args", lambda: args)
  monkeypatch.setattr(evaluation_cli, "MambaForestSegmenter", PredictionModel)
  monkeypatch.setattr(data_lib, "make_dataset", lambda *a, **kw: dataset)
  with pytest.raises(ValueError, match="yielded no batches"):
    evaluation_cli.main()
  assert not (tmp_path / "test_metrics.json").exists()
