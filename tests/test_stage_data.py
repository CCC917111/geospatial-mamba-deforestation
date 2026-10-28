"""Selected shard staging preserves immutable data identities across sessions."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from mamba_forest import stage_data


def _settings(tmp_path):
  source = tmp_path / "source"
  source.mkdir()
  for split in ("train", "validation", "test"):
    for index in range(4):
      name = f"forty_v1-{split}.tfrecord-{index:05d}-of-00004"
      (source / name).write_bytes((split + str(index)).encode() * 7)
  return {"source_root": source, "cache_root": tmp_path / "cache",
          "manifest_path": tmp_path / "persistent/selected-data.json",
          "total_shards": 4, "train_shards": 2, "val_shards": 1,
          "test_shards": 2, "selection_seed": 2026}


def test_import_and_default_planning_are_framework_free(tmp_path):
  source = Path(stage_data.__file__).resolve().parents[1]
  code = """import sys
from mamba_forest.stage_data import selected_files
files=selected_files()
assert len(files)==20
assert all(item['filename'].endswith('-of-01024') for item in files)
assert 'tensorflow' not in sys.modules
"""
  subprocess.run([sys.executable, "-c", code], check=True,
                 env={**os.environ, "PYTHONPATH": str(source)})
  assert not list(tmp_path.iterdir())


def test_planned_cohort_matches_real_dataset_selector():
  from mamba_forest import data

  files = stage_data.selected_files(
      "gs://forest_typology/forty_v1/1.0.0", total_shards=32,
      train_shards=4, val_shards=2, test_shards=3, selection_seed=2026)
  for split, budget in (("train", 4), ("validation", 2), ("test", 3)):
    expected = data.selected_shard_paths(
        split, budget, seed=2026, shuffle_shards=split == "train", total_shards=32)
    assert [item["source"] for item in files if item["split"] == split] == expected
  changed = stage_data.selected_files(
      total_shards=32, train_shards=4, val_shards=2, test_shards=3,
      selection_seed=2027)
  assert {item["filename"] for item in changed if item["split"] == "train"} != {
      item["filename"] for item in files if item["split"] == "train"}


def test_selected_local_files_copy_and_reuse_without_source_io(tmp_path, monkeypatch):
  args = _settings(tmp_path)
  manifest = stage_data.stage_dataset(**args)
  receipt = args["manifest_path"].read_bytes()
  assert manifest["schema_version"] == 1 and manifest["status"] == "ready"
  assert len(manifest["files"]) == 5
  assert manifest["config"]["selection_seed"] == 2026
  assert manifest["config"]["total_shards"] == 4
  assert {p.name for p in args["cache_root"].iterdir()} == {
      item["filename"] for item in manifest["files"]}
  for item in manifest["files"]:
    expected = (args["source_root"] / item["filename"]).read_bytes()
    assert (args["cache_root"] / item["filename"]).read_bytes() == expected
    assert item["bytes"] == len(expected)
    assert item["sha256"] == hashlib.sha256(expected).hexdigest()
  monkeypatch.setattr(stage_data, "_source_stat",
                      lambda path: pytest.fail("valid cache reread source"))
  monkeypatch.setattr(stage_data, "_open_source",
                      lambda path: pytest.fail("valid cache recopied source"))
  assert stage_data.stage_dataset(**args) == manifest
  assert args["manifest_path"].read_bytes() == receipt


@pytest.mark.parametrize("damage", ["missing", "short", "same_size"])
def test_missing_or_corrupt_cache_is_repaired_from_same_original(tmp_path, damage):
  args = _settings(tmp_path)
  manifest = stage_data.stage_dataset(**args)
  receipt = args["manifest_path"].read_bytes()
  item = manifest["files"][0]
  cached = args["cache_root"] / item["filename"]
  if damage == "missing":
    cached.unlink()
  else:
    cached.write_bytes(b"x" * (item["bytes"] if damage == "same_size" else 1))
  assert stage_data.stage_dataset(**args) == manifest
  assert hashlib.sha256(cached.read_bytes()).hexdigest() == item["sha256"]
  assert args["manifest_path"].read_bytes() == receipt
  assert not list(args["cache_root"].glob("*.partial"))


@pytest.mark.parametrize("same_size", [True, False])
def test_changed_source_cannot_redefine_existing_cohort(tmp_path, same_size):
  args = _settings(tmp_path)
  manifest = stage_data.stage_dataset(**args)
  receipt = args["manifest_path"].read_bytes()
  item = manifest["files"][0]
  cached = args["cache_root"] / item["filename"]
  cached.write_bytes(b"damaged cache")
  source = args["source_root"] / item["filename"]
  source.write_bytes(b"z" * (item["bytes"] if same_size else item["bytes"] + 1))
  with pytest.raises(ValueError, match="Source shard changed"):
    stage_data.stage_dataset(**args)
  assert cached.read_bytes() == b"damaged cache"
  assert args["manifest_path"].read_bytes() == receipt
  assert not list(args["cache_root"].glob("*.partial"))


@pytest.mark.parametrize("change", ["config", "partial", "filename", "sha", "bytes"])
def test_invalid_receipts_are_rejected_without_overwrite(tmp_path, change):
  args = _settings(tmp_path)
  manifest = stage_data.stage_dataset(**args)
  if change == "config":
    manifest["config"]["selection_seed"] = 42
  elif change == "partial":
    manifest["status"] = "copying"
  elif change == "filename":
    manifest["files"][0]["filename"] = "../outside"
  elif change == "sha":
    manifest["files"][0]["sha256"] = "not a hash"
  else:
    manifest["files"][0]["bytes"] = -1
  args["manifest_path"].write_text(json.dumps(manifest))
  original = args["manifest_path"].read_bytes()
  with pytest.raises(ValueError):
    stage_data.stage_dataset(**args)
  assert args["manifest_path"].read_bytes() == original


def test_changed_config_cannot_reuse_or_replace_receipt(tmp_path):
  args = _settings(tmp_path)
  stage_data.stage_dataset(**args)
  receipt = args["manifest_path"].read_bytes()
  with pytest.raises(ValueError, match="different config"):
    stage_data.stage_dataset(**{**args, "selection_seed": 2027})
  assert args["manifest_path"].read_bytes() == receipt


def test_total_byte_budget_is_checked_before_copying(tmp_path):
  args = _settings(tmp_path)
  with pytest.raises(ValueError, match="max-bytes"):
    stage_data.stage_dataset(**args, max_bytes=1)
  assert not args["cache_root"].exists()
  assert not args["manifest_path"].exists()


def test_free_space_failure_preserves_receipt_and_cleans_partial(
        tmp_path, monkeypatch):
  args = _settings(tmp_path)
  monkeypatch.setattr(stage_data.shutil, "disk_usage",
                      lambda path: SimpleNamespace(free=1))
  with pytest.raises(ValueError, match="disk space"):
    stage_data.stage_dataset(**args)
  assert not args["manifest_path"].exists()
  assert not list(args["cache_root"].iterdir())


def test_interrupted_copy_preserves_prior_receipt_and_cached_file(
        tmp_path, monkeypatch):
  args = _settings(tmp_path)
  manifest = stage_data.stage_dataset(**args)
  receipt = args["manifest_path"].read_bytes()
  cached = args["cache_root"] / manifest["files"][0]["filename"]
  cached.write_bytes(b"old damaged file")

  class Interrupted:
    def read(self, size):
      raise KeyboardInterrupt()

  @contextmanager
  def interrupted(path):
    yield Interrupted()

  monkeypatch.setattr(stage_data, "_open_source", interrupted)
  with pytest.raises(KeyboardInterrupt):
    stage_data.stage_dataset(**args)
  assert cached.read_bytes() == b"old damaged file"
  assert args["manifest_path"].read_bytes() == receipt
  assert not list(args["cache_root"].glob("*.partial"))


def test_unsafe_cache_symlink_is_rejected(tmp_path):
  args = _settings(tmp_path)
  manifest = stage_data.stage_dataset(**args)
  cached = args["cache_root"] / manifest["files"][0]["filename"]
  cached.unlink()
  unrelated = tmp_path / "unrelated"
  unrelated.write_bytes(b"leave alone")
  cached.symlink_to(unrelated)
  with pytest.raises(ValueError, match="safe regular file"):
    stage_data.stage_dataset(**args)
  assert unrelated.read_bytes() == b"leave alone"


def test_space_preflight_checks_whole_cohort_before_copy(tmp_path, monkeypatch):
  args = _settings(tmp_path)
  expected = stage_data.selected_files(
      args["source_root"], total_shards=4, train_shards=2,
      val_shards=1, test_shards=2, selection_seed=2026)
  sizes = [(args["source_root"] / item["filename"]).stat().st_size
           for item in expected]
  # Every individual shard fits, but the whole new cohort does not.
  monkeypatch.setattr(stage_data.shutil, "disk_usage",
                      lambda path: SimpleNamespace(free=max(sizes)))
  monkeypatch.setattr(stage_data, "_open_source",
                      lambda path: pytest.fail("copied before full space check"))
  with pytest.raises(ValueError, match="disk space"):
    stage_data.stage_dataset(**args)
  assert not args["manifest_path"].exists()
  assert not list(args["cache_root"].iterdir())


def test_initial_source_change_cannot_publish_a_ready_receipt(tmp_path, monkeypatch):
  args = _settings(tmp_path)
  original_stat = stage_data._source_stat
  calls = {}

  def changed(path):
    size, marker = original_stat(path)
    calls[path] = calls.get(path, 0) + 1
    return size, marker + (1 if calls[path] > 2 else 0)

  monkeypatch.setattr(stage_data, "_source_stat", changed)
  with pytest.raises(ValueError, match="changed during copy"):
    stage_data.stage_dataset(**args)
  assert not args["manifest_path"].exists()
  assert not list(args["cache_root"].glob("*.partial"))


@pytest.mark.parametrize("arguments", [
    {"train_shards": 0}, {"val_shards": True}, {"total_shards": 1.5},
    {"train_shards": 5}, {"selection_seed": -1}, {"selection_seed": 2**32},
    {"max_bytes": 0}, {"source_root": "https://invalid.example/data"},
])
def test_invalid_settings_fail_before_creating_output(tmp_path, arguments):
  args = {**_settings(tmp_path), **arguments}
  with pytest.raises(ValueError):
    stage_data.stage_dataset(**args)
  assert not args["cache_root"].exists()
  assert not args["manifest_path"].exists()


def test_local_cli_writes_complete_receipt(tmp_path, capsys):
  args = _settings(tmp_path)
  manifest = stage_data.main([
      "--source-root", str(args["source_root"]),
      "--cache-root", str(args["cache_root"]),
      "--manifest", str(args["manifest_path"]), "--total-shards", "4",
      "--train-shards", "2", "--val-shards", "1", "--test-shards", "2"])
  assert manifest["status"] == "ready"
  assert '"files": 5' in capsys.readouterr().out
