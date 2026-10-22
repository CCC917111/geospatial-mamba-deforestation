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
from mamba_forest.model import MambaForestSegmenter  # noqa: E402
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


def test_non_finite_predictions_are_rejected_before_confusion_updates():
  model = PredictionModel()
  model.offset.assign(float("nan"))
  with pytest.raises(tf.errors.InvalidArgumentError,
                     match="evaluation predictions contain non-finite"):
    evaluate(model, _dataset(*_examples()), 4)


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
      gcs_root=str(tmp_path / "records"), test_shards=2, total_shards=4, batch_size=2,
      d_model=8, dropout=0.0, d_state_temporal=2, d_state_spatial=2,
      bottleneck_channels=8, output_dir=str(tmp_path))


@pytest.mark.parametrize("flag", ["--data-root", "--gcs-root"])
def test_evaluation_data_root_alias_and_shard_count(flag, monkeypatch):
  monkeypatch.setattr(sys, "argv", [
      "mamba_forest.evaluate", "--weights", "local.weights.h5", flag,
      "local-records", "--total-shards", "4"])
  args = evaluation_cli.parse_args()
  assert args.gcs_root == "local-records"
  assert args.total_shards == 4


def test_cli_retains_recomputable_metrics_and_checkpoint_provenance(
      tmp_path, monkeypatch):
  args = _cli_args(tmp_path)
  dataset = _dataset(*_examples(), num_classes=data_lib.NUM_CLASSES)
  model = PredictionModel()
  model(data_lib.split_inputs_labels(next(iter(dataset)))[0])
  model.save_weights(args.weights)
  monkeypatch.setattr(evaluation_cli, "parse_args", lambda: args)
  monkeypatch.setattr(evaluation_cli, "MambaForestSegmenter", PredictionModel)
  dataset_calls = []

  def make_dataset(*a, **kw):
    dataset_calls.append((a, kw))
    return dataset

  monkeypatch.setattr(data_lib, "make_dataset", make_dataset)
  evaluation_cli.main()

  metrics = json.loads((tmp_path / "test_metrics.json").read_text())
  assert dataset_calls == [(("test", 2), {
      "batch_size": 2, "training": False, "gcs_root": args.gcs_root,
      "total_shards": 4,
  })]
  assert metrics["weights_sha256"] == hashlib.sha256(
      (tmp_path / "best.weights.h5").read_bytes()).hexdigest()
  assert metrics["gcs_root"] == args.gcs_root
  assert metrics["training_config_path"] is None
  assert metrics["split"] == "test"
  assert metrics["total_shards_per_split"] == 4
  assert metrics["shard_paths"] == data_lib.selected_shard_paths(
      "test", args.test_shards, gcs_root=args.gcs_root, total_shards=4)
  assert all(path.endswith("-of-00004") for path in metrics["shard_paths"])
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


def test_evaluation_config_errors_fail_before_dataset_io(tmp_path, monkeypatch):
  config_path = tmp_path / "config.json"
  config_path.write_text("{malformed JSON")
  monkeypatch.setattr(sys, "argv", [
      "mamba_forest.evaluate", "--weights", str(tmp_path / "best.weights.h5")])

  def unexpected_io(*args, **kwargs):
    raise AssertionError("configuration errors must fail before dataset I/O")

  monkeypatch.setattr(data_lib, "make_dataset", unexpected_io)
  with pytest.raises(SystemExit) as error:
    evaluation_cli.main()
  assert error.value.code == 2


def test_evaluation_sidecar_does_not_restore_data_or_precision(tmp_path, monkeypatch):
  config_path = tmp_path / "config.json"
  config_path.write_text(json.dumps({
      "d_model": 6, "dropout": 0.0, "d_state_temporal": 2,
      "d_state_spatial": 3, "bottleneck_channels": 8,
      "gcs_root": "training-only-records", "test_shards": 1,
      "total_shards": 3, "batch_size": 1, "no_mixed_precision": False,
  }))
  monkeypatch.setattr(sys, "argv", [
      "mamba_forest.evaluate", "--weights", str(tmp_path / "best.weights.h5")])
  before_policy = tf.keras.mixed_precision.global_policy().name
  args = evaluation_cli.parse_args()
  assert args.d_model == 6
  assert args.dropout == 0.0
  assert args.training_config_path == str(config_path)
  assert args.gcs_root == data_lib.GCS_ROOT
  assert args.test_shards == 64
  assert args.total_shards == data_lib.TOTAL_SHARDS
  assert args.batch_size == 16
  assert tf.keras.mixed_precision.global_policy().name == before_policy


def test_cli_reloads_nondefault_checkpoint_using_sidecar_and_local_records(
      tmp_path, monkeypatch):
  shapes = {
      "s2": (4, 16, 16, 10), "elevation": (16, 16, 3), "climate": (4, 14),
      "segmentation_labels": (16, 16),
  }
  description = {
      key: tf.io.FixedLenFeature(
          [int(np.prod(shape))],
          tf.int64 if key == "segmentation_labels" else tf.float32)
      for key, shape in shapes.items()
  }
  monkeypatch.setattr(data_lib, "FEATURE_SHAPES", shapes)
  monkeypatch.setattr(data_lib, "_FEATURE_DESCRIPTION", description)
  records_root = tmp_path / "records"
  records_root.mkdir()
  features = {}
  rng = np.random.default_rng(42)
  for key, shape in shapes.items():
    count = int(np.prod(shape))
    if key == "segmentation_labels":
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=np.arange(count) % data_lib.NUM_CLASSES))
    else:
      feature = tf.train.Feature(
          float_list=tf.train.FloatList(value=rng.uniform(0, 1000, count)))
    features[key] = feature
  example = tf.train.Example(features=tf.train.Features(feature=features))
  record_path = data_lib.shard_paths("test", str(records_root), total_shards=1)[0]
  with tf.io.TFRecordWriter(record_path) as writer:
    writer.write(example.SerializeToString())

  config = {"d_model": 6, "dropout": 0.0, "d_state_temporal": 2,
            "d_state_spatial": 3, "bottleneck_channels": 8}
  model = MambaForestSegmenter(
      num_classes=data_lib.NUM_CLASSES, d_model=config["d_model"],
      dropout_rate=config["dropout"], d_state_temporal=config["d_state_temporal"],
      d_state_spatial=config["d_state_spatial"],
      bottleneck_channels=config["bottleneck_channels"])
  dataset = data_lib.make_dataset(
      "test", 1, batch_size=1, training=False,
      gcs_root=str(records_root), total_shards=1)
  inputs, labels = data_lib.split_inputs_labels(next(iter(dataset)))
  prediction = model(inputs, training=False)
  expected_confusion = tf.math.confusion_matrix(
      tf.reshape(tf.argmax(labels, axis=-1), [-1]),
      tf.reshape(tf.argmax(prediction, axis=-1), [-1]),
      num_classes=data_lib.NUM_CLASSES, dtype=tf.int64).numpy()
  weights_path = tmp_path / "best.weights.h5"
  model.save_weights(weights_path)
  config_path = tmp_path / "config.json"
  config_path.write_text(json.dumps(config))
  monkeypatch.setattr(sys, "argv", [
      "mamba_forest.evaluate", "--weights", str(weights_path),
      "--data-root", str(records_root), "--test-shards", "1",
      "--total-shards", "1", "--batch-size", "1"])
  evaluation_cli.main()

  metrics = json.loads((tmp_path / "test_metrics.json").read_text())
  assert metrics["training_config_path"] == str(config_path)
  assert metrics["shard_paths"] == [record_path]
  assert metrics["num_pixels"] == 16 * 16
  assert metrics["model_config"]["dropout_rate"] == config["dropout"]
  for key in ("d_model", "d_state_temporal", "d_state_spatial", "bottleneck_channels"):
    assert metrics["model_config"][key] == config[key]
  np.testing.assert_array_equal(metrics["confusion_matrix"], expected_confusion)
  assert metrics["pixel_accuracy"] == pytest.approx(
      np.trace(expected_confusion) / expected_confusion.sum())
