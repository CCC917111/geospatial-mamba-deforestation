"""Local synthetic-schema, fresh-output, and failure-diagnostic demo checks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mamba_forest import data, demo  # noqa: E402
from mamba_forest.constants import FEATURE_SHAPES, NUM_CLASSES  # noqa: E402


def test_generated_splits_use_production_schema_and_keep_test_npz_consistent(tmp_path):
  manifest = demo.generate_synthetic_data(
      tmp_path, 42, {"train": 2, "validation": 1, "test": 1})
  identities, hashes, optical_hashes = [], [], []
  for split, expected_count in (("train", 2), ("validation", 1), ("test", 1)):
    entry = manifest["splits"][split]
    assert entry["examples"] == expected_count
    assert entry["path"].endswith(f"-{split}.tfrecord-00000-of-00001")
    records = list(tf.data.TFRecordDataset(entry["path"]).as_numpy_iterator())
    assert len(records) == expected_count
    for index, record in enumerate(records):
      raw = tf.train.Example.FromString(record)
      sample_id = raw.features.feature["sample_id"].bytes_list.value[0].decode()
      assert sample_id == f"synthetic_{split}_{index:03d}"
      identities.append(sample_id)
      hashes.append(entry["samples"][index]["record_sha256"])
      parsed = data.parse_example(tf.constant(record))
      assert parsed["segmentation_labels"].dtype == tf.int64
      assert parsed["segmentation_labels"].shape == (128, 128)
      for key, shape in FEATURE_SHAPES.items():
        assert parsed[key].shape == shape
        assert np.isfinite(parsed[key].numpy()).all()
      optical_hashes.append(hashlib.sha256(parsed["s2"].numpy().tobytes()).hexdigest())
      labels = parsed["segmentation_labels"].numpy()
      np.testing.assert_array_equal(np.unique(labels), np.arange(NUM_CLASSES))
      processed = data.preprocess(parsed, training=False)
      np.testing.assert_array_equal(processed["labels"].numpy().argmax(-1), labels)
      if split == "test":
        with np.load(manifest["test_tiles"][index], allow_pickle=False) as tile:
          for key in FEATURE_SHAPES:
            np.testing.assert_array_equal(tile[key], parsed[key].numpy())
  assert len(set(identities)) == len(identities)
  assert len(set(hashes)) == len(hashes)
  assert len(set(optical_hashes)) == len(optical_hashes)
  retained = json.loads((tmp_path / "data" / "manifest.json").read_text())
  assert retained["data_kind"] == "synthetic"
  assert "not satellite observations" in retained["scope"]


def test_identical_seed_reproduces_records_without_reusing_split_examples(tmp_path):
  roots = [tmp_path / "first", tmp_path / "second"]
  manifests = []
  for root in roots:
    root.mkdir()
    manifests.append(demo.generate_synthetic_data(
        root, 7, {"train": 1, "validation": 1, "test": 1}))
  for split in ("train", "validation", "test"):
    assert manifests[0]["splits"][split]["file_sha256"] == (
        manifests[1]["splits"][split]["file_sha256"])
  assert len({entry["file_sha256"] for entry in manifests[0]["splits"].values()}) == 3


@pytest.mark.parametrize("existing_file", [False, True])
def test_existing_output_content_is_protected_before_any_demo_work(
      tmp_path, monkeypatch, existing_file):
  destination = tmp_path / "existing"
  if existing_file:
    destination.write_bytes(b"keep this existing file")
    protected = destination
  else:
    destination.mkdir()
    protected = destination / "user-notes.txt"
    protected.write_bytes(b"keep these user notes")
  before = protected.read_bytes()
  args = demo.parse_args(["--output-dir", str(destination)])

  def unexpected_work():
    pytest.fail("existing outputs must be rejected before source or data access")

  monkeypatch.setattr(demo, "_source_info", unexpected_work)
  with pytest.raises(FileExistsError, match="choose a new"):
    demo.run_demo(args)
  assert protected.read_bytes() == before
  if not existing_file:
    assert sorted(path.name for path in destination.iterdir()) == ["user-notes.txt"]


def test_default_output_is_unique_and_an_existing_empty_directory_is_allowed(
      tmp_path, monkeypatch):
  monkeypatch.chdir(tmp_path)
  first = demo.prepare_output_directory(None)
  second = demo.prepare_output_directory(None)
  assert first != second
  assert first.parent == second.parent == tmp_path / "runs"
  assert first.name.startswith("synthetic-demo-")
  empty = tmp_path / "empty"
  empty.mkdir()
  assert demo.prepare_output_directory(empty) == empty


@pytest.mark.parametrize("preset_caches", [False, True])
def test_failed_stage_retains_diagnostics_and_restores_runtime_environment(
      tmp_path, monkeypatch, preset_caches):
  destination = tmp_path / "failed-demo"
  args = demo.parse_args(["--output-dir", str(destination),
                          "--train-examples", "1", "--val-examples", "1",
                          "--test-examples", "1"])
  monkeypatch.setattr(demo, "_source_info", lambda: {"git_commit": "local-test-source"})
  for key in ("MPLCONFIGDIR", "XDG_CACHE_HOME"):
    if preset_caches:
      path = tmp_path / ("caller-" + key.lower())
      path.mkdir()
      monkeypatch.setenv(key, str(path))
    else:
      monkeypatch.delenv(key, raising=False)
  original_environment = dict(os.environ)

  def failed_stage(command, log_path, environment):
    assert environment["CUDA_VISIBLE_DEVICES"] == "-1"
    assert environment["TF_NUM_INTRAOP_THREADS"] == "2"
    assert environment["TF_NUM_INTEROP_THREADS"] == "2"
    for key in ("MPLCONFIGDIR", "XDG_CACHE_HOME"):
      assert Path(environment[key]).is_dir()
      if preset_caches:
        assert environment[key] == original_environment[key]
      else:
        assert Path(environment[key]).is_relative_to(destination / "runtime")
    assert command[1:3] == ["-m", "mamba_forest.train"]
    log_path.write_text("Synthetic failure: training process stopped.\n")
    raise RuntimeError(f"stage exited 7; see {log_path}")

  monkeypatch.setattr(demo, "_run_stage", failed_stage)
  with pytest.raises(RuntimeError, match="stage exited 7"):
    demo.run_demo(args)
  assert dict(os.environ) == original_environment
  summary = json.loads((destination / "summary.json").read_text())
  assert summary["status"] == "failed"
  assert summary["error"]["stage"] == "train"
  assert summary["stages"][0]["state"] == "failed"
  assert Path(summary["stages"][0]["log_path"]).read_text().startswith(
      "Synthetic failure")
  assert (destination / "data" / "manifest.json").is_file()
  assert summary["source"]["git_commit"] == "local-test-source"
  assert "real-data accuracy" in summary["scope"]


def test_orchestration_uses_real_entrypoints_and_checkpoint_sidecar_defaults(
      tmp_path, monkeypatch):
  destination = tmp_path / "mock-demo"
  args = demo.parse_args(["--output-dir", str(destination)])
  monkeypatch.setattr(demo, "_source_info", lambda: {"git_commit": "test-source"})
  monkeypatch.setattr(demo, "generate_synthetic_data",
                      lambda *args: {"test_tiles": []})
  calls = []

  def completed_stage(command, log_path, environment):
    calls.append(command)
    log_path.write_text("Mock subprocess completed; no model was trained.\n")

  monkeypatch.setattr(demo, "_run_stage", completed_stage)
  monkeypatch.setattr(demo, "_collect_results", lambda *args: {"completed_epochs": 2})
  monkeypatch.setattr(demo, "render_comparison",
                      lambda root, manifest: (root / "comparison.png").write_bytes(
                          b"mock"))
  original_environment = dict(os.environ)
  summary = demo.run_demo(args)
  assert dict(os.environ) == original_environment
  assert summary["status"] == "completed"
  assert [command[2] for command in calls] == [
      "mamba_forest.train", "mamba_forest.evaluate", "mamba_forest.predict"]
  assert calls[0][calls[0].index("--total-shards") + 1] == "1"
  assert calls[0][calls[0].index("--precision") + 1] == "float32"
  assert calls[0][calls[0].index("--epochs") + 1] == "2"
  assert calls[0][calls[0].index("--d-model") + 1] == "4"
  assert "--fixed-train-subset" in calls[0]
  assert "--d-model" not in calls[1] and "--d-model" not in calls[2]
  assert "--save-png" in calls[2]
  assert calls[2][-2:] == [
      "--timings", str(destination / "predictions" / "timings.json")]


@pytest.mark.parametrize("arguments", [
    ["--train-examples", "0"], ["--test-examples", "-1"], ["--epochs", "0"],
    ["--batch-size", "0"], ["--seed", "-1"],
])
def test_invalid_demo_budgets_fail_before_creating_output(arguments):
  with pytest.raises(SystemExit) as error:
    demo.parse_args(arguments)
  assert error.value.code == 2
