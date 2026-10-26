"""Training provenance helpers and a two-epoch local TFRecord smoke test."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mamba_forest import data, train  # noqa: E402
from mamba_forest.model import MambaForestSegmenter  # noqa: E402


def _subset_args(**overrides):
  args = dict(train_shards=2, val_shards=1, batch_size=2, seed=42,
              gcs_root="local-records", fixed_train_subset=False, total_shards=4)
  args.update(overrides)
  return argparse.Namespace(**args)


@pytest.mark.parametrize("flag", ["--data-root", "--gcs-root"])
def test_training_data_root_alias_and_shard_count(flag, monkeypatch):
  monkeypatch.setattr(sys, "argv", [
      "mamba_forest.train", flag, "local-records", "--total-shards", "4"])
  args = train.parse_args()
  assert args.gcs_root == "local-records"
  assert args.total_shards == 4


def test_fixed_and_rotating_manifests_match_dataset_epoch(monkeypatch):
  calls = []

  def make_dataset(*args, **kwargs):
    calls.append((args, kwargs))
    return "dataset"

  monkeypatch.setattr(data, "make_dataset", make_dataset)
  for fixed in (False, True):
    args = _subset_args(fixed_train_subset=fixed)
    assert train.build_train_dataset(args, epoch=3) == "dataset"
    manifest = train.training_epoch_manifest(args, epoch=4)
    shard_epoch = 0 if fixed else 3
    assert calls[-1] == (("train", 2), {
        "batch_size": 2, "training": True, "seed": 42,
        "gcs_root": "local-records", "epoch": shard_epoch, "total_shards": 4,
    })
    assert manifest["epoch"] == 4
    assert manifest["shard_epoch"] == shard_epoch
    assert manifest["training_completed"] is False
    assert manifest["paths"] == data.selected_shard_paths(
        "train", 2, seed=42, epoch=shard_epoch, shuffle_shards=True,
        gcs_root="local-records", total_shards=4)
    first = train.training_epoch_manifest(args, epoch=1)
    assert (first["paths"] == manifest["paths"]) == fixed


def test_split_manifest_counts_only_completed_unique_membership(tmp_path):
  path = str(tmp_path / "split_manifest.json")
  manifest = {
      "training_mode": "rotating",
      "epochs": [
          {"epoch": 1, "paths": ["a", "b"], "training_completed": True},
          {"epoch": 2, "paths": ["b", "c"], "training_completed": False},
      ],
  }
  train.write_split_manifest(path, manifest)
  with open(path, encoding="utf-8") as handle:
    saved = json.load(handle)
  assert saved["completed_unique_train_shards"] == 2
  assert saved["training_mode"] == "rotating"
  assert saved["epochs"][1]["training_completed"] is False
  assert not os.path.exists(path + ".tmp")

  manifest["epochs"][1]["training_completed"] = True
  train.write_split_manifest(path, manifest)
  with open(path, encoding="utf-8") as handle:
    saved = json.load(handle)
  assert saved["completed_unique_train_shards"] == 3
  assert saved["epochs"][1]["paths"] == ["b", "c"]


@pytest.mark.parametrize("checkpoint_name", ["best.weights.h5", "export.weights.h5"])
def test_existing_checkpoint_and_config_survive_rejected_training_rerun(
      tmp_path, monkeypatch, checkpoint_name):
  run = tmp_path / "run"
  run.mkdir()
  config_bytes = b'{"d_model": 4, "dropout": 0.1}'
  checkpoint_bytes = b"previous checkpoint contents"
  (run / "config.json").write_bytes(config_bytes)
  (run / checkpoint_name).write_bytes(checkpoint_bytes)
  monkeypatch.setattr(sys, "argv", [
      "train", "--output-dir", str(run), "--d-model", "8"])

  def unexpected_training_io(*args, **kwargs):
    pytest.fail("existing checkpoint must be protected before dataset access")

  monkeypatch.setattr(train, "build_datasets", unexpected_training_io)
  previous_policy = tf.keras.mixed_precision.global_policy().name
  with pytest.raises(FileExistsError, match="choose a new --output-dir"):
    train.main()
  assert (run / "config.json").read_bytes() == config_bytes
  assert (run / checkpoint_name).read_bytes() == checkpoint_bytes
  assert {path.name for path in run.iterdir()} == {"config.json", checkpoint_name}
  assert tf.keras.mixed_precision.global_policy().name == previous_policy


def _small_record(shapes, seed):
  rng = np.random.default_rng(seed)
  features = {}
  for name, shape in shapes.items():
    if name == "segmentation_labels":
      values = np.arange(np.prod(shape), dtype=np.int64) % data.NUM_CLASSES
      feature = tf.train.Feature(int64_list=tf.train.Int64List(value=values))
    else:
      values = rng.uniform(0, 1, size=shape).astype(np.float32)
      if name == "s2":
        values *= 10000
      feature = tf.train.Feature(
          float_list=tf.train.FloatList(value=values.ravel()))
    features[name] = feature
  return tf.train.Example(
      features=tf.train.Features(feature=features)).SerializeToString()


@pytest.mark.parametrize("model_variant", ["cnn_ssm", "cnn_only"])
def test_training_main_two_epochs_saves_reusable_checkpoints(
        tmp_path, monkeypatch, model_variant):
  shapes = {
      "s2": (4, 16, 16, 10), "elevation": (16, 16, 3),
      "climate": (4, 14), "segmentation_labels": (16, 16),
  }
  description = {
      key: tf.io.FixedLenFeature(
          [int(np.prod(shape))],
          tf.int64 if key == "segmentation_labels" else tf.float32)
      for key, shape in shapes.items()
  }
  monkeypatch.setattr(data, "FEATURE_SHAPES", shapes)
  monkeypatch.setattr(data, "_FEATURE_DESCRIPTION", description)
  root = str(tmp_path / "records")
  os.makedirs(root)
  output = tmp_path / "run"
  subset_args = _subset_args(gcs_root=root)
  expected = [train.training_epoch_manifest(subset_args, epoch)
              for epoch in (1, 2)]
  validation_paths = data.selected_shard_paths(
      "validation", 1, gcs_root=root, total_shards=subset_args.total_shards)
  paths = sorted({path for epoch in expected for path in epoch["paths"]}
                 | set(validation_paths))
  for index, path in enumerate(paths):
    with tf.io.TFRecordWriter(path) as writer:
      writer.write(_small_record(shapes, seed=index))

  monkeypatch.setattr(sys, "argv", [
      "mamba_forest.train", "--data-root", root, "--total-shards", "4",
      "--output-dir", str(output),
      "--epochs", "2", "--train-shards", "2", "--val-shards", "1",
      "--batch-size", "2", "--d-model", "4", "--d-state-temporal", "2",
      "--d-state-spatial", "2", "--bottleneck-channels", "8",
      "--dropout", "0", "--patience", "0",
      "--model-variant", model_variant,
  ])
  models = []

  def build_model(**kwargs):
    model = MambaForestSegmenter(**kwargs)
    models.append(model)
    return model

  original_visible_devices = tf.config.get_visible_devices

  def visible_devices(device_type=None):
    if device_type == "GPU":
      return []
    return original_visible_devices(device_type)

  def unexpected_loss_scaling(*args, **kwargs):
    pytest.fail("automatic CPU training must not construct LossScaleOptimizer")

  monkeypatch.setattr(train, "MambaForestSegmenter", build_model)
  monkeypatch.setattr(tf.config, "get_visible_devices", visible_devices)
  monkeypatch.setattr(tf.keras.mixed_precision, "LossScaleOptimizer",
                      unexpected_loss_scaling)
  original_evaluator = train.DatasetEvaluator
  selection_scores = [0.12345671, 0.12345674]
  evaluated_epochs = []
  validation_evaluators = []
  validation_trace_counts = []

  class CloseValidationEvaluator(original_evaluator):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      validation_evaluators.append(self)

    def __call__(self, *args, **kwargs):
      accuracy, mean_iou, _ = super().__call__(*args, **kwargs)
      validation_trace_counts.append(self.accumulate.experimental_get_tracing_count())
      score = selection_scores[len(evaluated_epochs)]
      evaluated_epochs.append(score)
      return accuracy, mean_iou, np.full(data.NUM_CLASSES, score)

  # Both scores round to the same six-decimal CSV value. The second epoch
  # must still be recoverable as the checkpoint selected during training.
  monkeypatch.setattr(train, "DatasetEvaluator", CloseValidationEvaluator)
  prior_policy = tf.keras.mixed_precision.global_policy().name
  try:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    train.main()
    assert len(validation_evaluators) == 1
    assert len(validation_trace_counts) == 2
    assert validation_trace_counts[0] == validation_trace_counts[1]
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["precision"] == "auto"
    assert config["resolved_precision"] == "float32"
    assert config["model_variant"] == model_variant
    assert models[0].compute_dtype == "float32"
    with open(output / "split_manifest.json", encoding="utf-8") as handle:
      manifest = json.load(handle)
    assert manifest["training_mode"] == "rotating"
    assert manifest["seed"] == subset_args.seed
    assert manifest["validation_paths"] == validation_paths
    assert [entry["paths"] for entry in manifest["epochs"]] == [
        entry["paths"] for entry in expected]
    assert all(entry["training_completed"] for entry in manifest["epochs"])
    assert manifest["completed_unique_train_shards"] == len(paths) - 1
    assert manifest["total_shards_per_split"] == 4
    assert all(path.endswith("-of-00004") for path in paths)

    with open(output / "training_results.csv", encoding="utf-8") as handle:
      rows = list(csv.DictReader(handle))
    assert [int(row["epoch"]) for row in rows] == [1, 2]
    assert [float(row["overall_f1"]) for row in rows] == [
        float(np.mean(np.full(data.NUM_CLASSES, score)))
        for score in selection_scores]
    selection = json.loads(
        (output / "checkpoint_selection.json").read_text(encoding="utf-8"))
    assert selection["epoch"] == 2
    assert selection["score"] == float(rows[1]["overall_f1"])
    assert selection["criterion"] == "validation_overall_f1"
    assert selection["weights"] == "best.weights.h5"
    assert selection["weights_sha256"] == train.weights_sha256(
        output / "best.weights.h5")
    for row in rows:
      assert all(np.isfinite(float(value)) for value in row.values())
      assert float(row["train_examples"]) == 2
      assert float(row["train_seconds"]) > 0
      assert float(row["validation_seconds"]) > 0
      assert np.isclose(
          float(row["train_examples_per_second"]),
          2 / float(row["train_seconds"]), rtol=1e-4)

    validation = next(iter(data.make_dataset(
        "validation", 1, batch_size=1, gcs_root=root, total_shards=4)))
    inputs, _ = data.split_inputs_labels(validation)
    reference = models[0](inputs, training=False).numpy()
    for name in ("last.weights.h5", "best.weights.h5"):
      restored = MambaForestSegmenter(
          num_classes=data.NUM_CLASSES, d_model=4, dropout_rate=0,
          d_state_temporal=2, d_state_spatial=2, bottleneck_channels=8,
          model_variant=model_variant)
      restored(inputs, training=False)
      restored.load_weights(str(output / name))
      predictions = restored(inputs, training=False).numpy()
      assert np.all(np.isfinite(predictions))
      assert np.allclose(predictions.sum(axis=-1), 1, atol=1e-5)
      assert np.allclose(predictions, reference, rtol=1e-5, atol=1e-6)
  finally:
    tf.keras.mixed_precision.set_global_policy(prior_policy)
