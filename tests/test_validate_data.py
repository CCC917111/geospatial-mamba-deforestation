"""Local raw-record preflight checks, including fail-fast GPU gating."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mamba_forest import data  # noqa: E402
from mamba_forest import validate_data  # noqa: E402


def _args(tmp_path, **overrides):
  config = dict(gcs_root=str(tmp_path), train_shards=1, val_shards=1,
                test_shards=1, total_shards=1, seed=42, require_gpu=False,
                output=str(tmp_path / "preflight.json"))
  config.update(overrides)
  return argparse.Namespace(**config)


def _record(label=2, sample_id=None, bad_feature=None, nonfinite=False):
  features = {}
  for name, shape in data.FEATURE_SHAPES.items():
    count = int(np.prod(shape))
    if name == "segmentation_labels":
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=[label] * count))
    else:
      values = np.zeros(count, dtype=np.float32)
      if nonfinite and name == "s2":
        values[0] = np.nan
      feature = tf.train.Feature(float_list=tf.train.FloatList(value=values))
    features[name] = feature
  if sample_id is not None:
    features["sample_id"] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[sample_id.encode()]))
    features["latitude"] = tf.train.Feature(
        float_list=tf.train.FloatList(value=[22.5]))
    features["longitude"] = tf.train.Feature(
        float_list=tf.train.FloatList(value=[114.25]))
  if bad_feature is not None:
    features[bad_feature] = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[b"wrong feature kind"]))
  example = tf.train.Example(features=tf.train.Features(feature=features))
  return example.SerializeToString()


def _write_splits(args, **record_options):
  selected = {}
  budgets = (("train", args.train_shards), ("validation", args.val_shards),
             ("test", args.test_shards))
  for split, budget in budgets:
    paths = data.selected_shard_paths(
        split, budget, seed=args.seed, epoch=0, shuffle_shards=split == "train",
        gcs_root=args.gcs_root, total_shards=args.total_shards)
    selected[split] = paths
    options = dict(sample_id=f"{split}-sample")
    options.update(record_options)
    with tf.io.TFRecordWriter(paths[0]) as writer:
      writer.write(_record(**options))
      # A preflight is deliberately limited to the first record.
      writer.write(b"malformed second record, not inspected")
  return selected


def test_gpu_gate_fails_before_any_record_io(tmp_path, monkeypatch):
  args = _args(tmp_path, require_gpu=True, gcs_root="gs://not-opened")
  monkeypatch.setattr(tf.config, "list_physical_devices", lambda: [])

  def unexpected_io(*args, **kwargs):
    raise AssertionError("GPU gate must run before record I/O")

  monkeypatch.setattr(tf.data, "TFRecordDataset", unexpected_io)
  report = validate_data.run_preflight(args)
  assert report["status"] == "failed"
  assert report["splits"] == {}
  assert report["errors"][0]["stage"] == "gpu"


def test_three_split_preflight_records_raw_schema_and_pipeline_provenance(tmp_path):
  args = _args(tmp_path, train_shards=2, total_shards=3)
  selected = _write_splits(args)
  report = validate_data.run_preflight(args)
  assert report["status"] == "passed"
  assert report["errors"] == []
  assert "does not establish geographic isolation" in report["scope"]
  assert report["runtime"]["tensorflow_version"] == tf.__version__
  assert report["runtime"]["devices"]
  for split, inspected in report["splits"].items():
    assert inspected["status"] == "passed"
    assert inspected["selected_shard_paths"] == selected[split]
    assert inspected["path"] == selected[split][0]
    assert inspected["record_index"] == 0
    assert inspected["raw_features"]["s2"] == {
        "kind": "float_list", "length": 4 * 128 * 128 * 10}
    assert inspected["raw_features"]["segmentation_labels"] == {
        "kind": "int64_list", "length": 128 * 128}
    assert inspected["tensors"]["parsed"]["segmentation_labels"] == {
        "shape": [128, 128], "dtype": "int64"}
    assert inspected["tensors"]["normalized"]["s2"] == {
        "shape": [4, 128, 128, 10], "dtype": "float32"}
    assert inspected["tensors"]["encoded"]["labels"] == {
        "shape": [128, 128, data.NUM_CLASSES], "dtype": "float32"}
    assert all(inspected["finite_inputs"].values())
    assert inspected["labels"]["min"] == inspected["labels"]["max"] == 2
    assert inspected["labels"]["num_pixels"] == 128 * 128
    assert inspected["labels"]["class_support"]["Planted Forest"] == 128 * 128
    assert inspected["metadata"] == {
        "sample_id": [f"{split}-sample"], "latitude": [22.5], "longitude": [114.25]}


def test_malformed_raw_feature_retains_schema_error_and_exact_path(tmp_path):
  args = _args(tmp_path)
  paths = _write_splits(args, bad_feature="s2")
  report = validate_data.run_preflight(args)
  assert report["status"] == "failed"
  assert list(report["splits"]) == ["train"]
  assert report["splits"]["train"]["raw_features"]["s2"] == {
      "kind": "bytes_list", "length": 1}
  assert report["errors"][0]["path"] == paths["train"][0]
  assert report["errors"][0]["type"] == "InvalidArgumentError"
  assert "s2" in report["errors"][0]["message"]


@pytest.mark.parametrize("record_options, message", [
    ({"nonfinite": True}, "non-finite normalized inputs"),
    ({"label": data.NUM_CLASSES}, "segmentation_labels"),
])
def test_invalid_values_fail_preflight(tmp_path, record_options, message):
  args = _args(tmp_path)
  _write_splits(args, **record_options)
  report = validate_data.run_preflight(args)
  assert report["status"] == "failed"
  assert message in report["errors"][0]["message"]


def test_duplicate_inspected_sample_ids_across_splits_fail(tmp_path):
  args = _args(tmp_path)
  _write_splits(args, sample_id="duplicate-tile")
  report = validate_data.run_preflight(args)
  assert report["status"] == "failed"
  assert report["splits"]["train"]["status"] == "passed"
  assert report["splits"]["validation"]["status"] == "failed"
  assert "duplicate inspected sample ID" in report["errors"][0]["message"]


def test_cli_writes_failure_json_and_exits_nonzero(tmp_path, monkeypatch):
  args = _args(tmp_path, require_gpu=True)
  monkeypatch.setattr(validate_data, "parse_args", lambda: args)
  monkeypatch.setattr(tf.config, "list_physical_devices", lambda: [])
  with pytest.raises(SystemExit) as error:
    validate_data.main()
  assert error.value.code == 1
  report = json.loads((tmp_path / "preflight.json").read_text())
  assert report["status"] == "failed"


def test_cli_writes_success_json(tmp_path, monkeypatch):
  args = _args(tmp_path)
  _write_splits(args)
  monkeypatch.setattr(validate_data, "parse_args", lambda: args)
  validate_data.main()
  report = json.loads((tmp_path / "preflight.json").read_text())
  assert report["status"] == "passed"
