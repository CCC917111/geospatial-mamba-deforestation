"""Retained research reports validate bytes and publish numeric class IDs."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest

from mamba_forest import research_report
from mamba_forest.constants import GCS_ROOT


SPEC = importlib.util.spec_from_file_location(
    "comparison_artifact_fixtures", Path(__file__).with_name("test_comparison.py"))
FIXTURES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIXTURES)


def _json(path, payload):
  Path(path).write_text(json.dumps(payload), encoding="utf-8")


def _research_runs(root, seeds=(42, 43, 44), selection_seed=2026,
                   data_kind="synthetic", total_shards=4, source_root=None):
  cache = root / "records"
  source = root / "original-records"
  cache.mkdir()
  source.mkdir()
  names = {split: [f"forty_v1-{split}.tfrecord-{index:05d}-of-{total_shards:05d}"
                   for index in range(total_shards)]
           for split in ("train", "validation", "test")}
  selected = {"train": random.Random(selection_seed).sample(names["train"], 2),
              "validation": names["validation"][:1], "test": names["test"][:1]}
  records = []
  for split, filenames in selected.items():
    for filename in filenames:
      payload = ("synthetic fixture " + filename).encode()
      (cache / filename).write_bytes(payload)
      (source / filename).write_bytes(payload)
      records.append({"split": split, "filename": filename,
                      "source": (source_root or str(source)) + "/" + filename,
                      "bytes": len(payload),
                      "sha256": hashlib.sha256(payload).hexdigest()})
  for index, seed in enumerate(seeds):
    for variant, correct in (("cnn_ssm", 7 + index % 3), ("cnn_only", 6)):
      directory = FIXTURES._write_run(root, variant, seed, correct=correct)
      config = json.loads((directory / "config.json").read_text())
      config.update(selection_seed=selection_seed, gcs_root=str(cache),
                    total_shards=total_shards)
      _json(directory / "config.json", config)
      manifest = json.loads((directory / "split_manifest.json").read_text())
      manifest.update(gcs_root=str(cache), selection_seed=selection_seed,
                      total_shards_per_split=total_shards,
                      effective_selection_seed=selection_seed,
                      validation_paths=[str(cache / name)
                                        for name in selected["validation"]])
      for epoch in manifest["epochs"]:
        epoch["paths"] = [str(cache / name) for name in selected["train"]]
      _json(directory / "split_manifest.json", manifest)
      metrics = json.loads((directory / "test/test_metrics.json").read_text())
      metrics.update(gcs_root=str(cache),
                     total_shards_per_split=total_shards,
                     shard_paths=[str(cache / name) for name in selected["test"]])
      _json(directory / "test/test_metrics.json", metrics)
      (directory / "last.weights.h5").write_bytes(b"last fixture weights")
      (directory / "train_wall_seconds.txt").write_text("9\n")
      (directory / "test_wall_seconds.txt").write_text("1\n")
  manifest_path = root / "selected_data.json"
  _json(manifest_path, {
      "schema_version": 1, "status": "ready", "config": {
          "source_root": source_root or str(source), "cache_root": str(cache),
          "total_shards": total_shards,
          "train_shards": 2, "val_shards": 1, "test_shards": 1,
          "selection_seed": selection_seed}, "files": records})
  _json(root / "research_spec.json", {"data_kind": data_kind})
  return manifest_path


def test_real_comparator_gates_numeric_scores_and_retained_checkpoint_bytes(tmp_path):
  data = _research_runs(tmp_path)
  session = tmp_path / "sessions/session-1"
  session.mkdir(parents=True)
  _json(session / "runtime.json", {"tensorflow_version": "2.21.0"})
  _json(tmp_path / "execution.json", {"state": "running"})
  report = research_report.build_report(tmp_path, data_manifest=data)
  assert report["status"] == "software_check_complete"
  assert report["seed_count"] == 3 and report["research_pilot_multiseed"]
  assert report["class_ids"] == list(range(9))
  assert report["coverage"]["published_shards_per_split"] is None
  assert not report["coverage"]["full_published_test_split"]
  metrics = report["summary"]["cnn_ssm"]["metrics_percent"]
  assert metrics["macro_f1_all_ids"] == {"mean": 80, "sample_sd": 10}
  assert metrics["pixel_accuracy"] == {"mean": 80, "sample_sd": 10}
  delta = report["summary"]["paired_delta_percentage_points"]
  assert delta["macro_f1_all_ids"] == {"mean": 20, "sample_sd": 10}
  assert report["pairs"][0]["cnn_ssm"]["test_pixels"] == 90
  assert [x["id"] for x in report["pairs"][0]["cnn_ssm"]["per_numeric_class"]] == list(
      range(9))
  serialized = json.dumps(report)
  for unsupported in ("forests_f1", "natural_forest_f1", "Natural Forest",
                      "Tree Crops"):
    assert unsupported not in serialized
  assert json.loads((tmp_path / "research_report.json").read_text()) == report
  manifest = json.loads((tmp_path / "checkpoint_manifest.json").read_text())
  assert manifest["selected_checkpoint_count"] == 6
  assert len(manifest["checkpoints"]) == 12
  assert all(record["path"] != "execution.json" for record in manifest["files"])
  assert any(record["path"].endswith("runtime.json") for record in manifest["files"])
  for record in manifest["files"]:
    path = Path(record["path"]) if record["external"] else tmp_path / record["path"]
    assert path.stat().st_size == record["bytes"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
  retained_hash = hashlib.sha256(
      (tmp_path / "checkpoint_manifest.json").read_bytes()).hexdigest()
  assert retained_hash == report["retention_manifest"]["sha256"]


@pytest.mark.parametrize("change", ["missing_manifest", "changed_bytes", "cohort",
                                    "wrong_cache", "missing_last", "changed_best"])
def test_incomplete_or_changed_research_evidence_never_writes_success(tmp_path, change):
  data = _research_runs(tmp_path)
  manifest = json.loads(data.read_text())
  if change == "missing_manifest":
    data = None
  elif change == "changed_bytes":
    (tmp_path / "records" / manifest["files"][0]["filename"]).write_bytes(b"changed")
  elif change == "cohort":
    manifest["files"].pop()
    _json(data, manifest)
  elif change == "wrong_cache":
    manifest["config"]["cache_root"] = "/another/cache"
    _json(data, manifest)
  elif change == "missing_last":
    (tmp_path / "cnn_only_44/last.weights.h5").unlink()
  elif change == "changed_best":
    (tmp_path / "cnn_ssm_42/best.weights.h5").write_bytes(b"changed")
  with pytest.raises((ValueError, FileNotFoundError)):
    research_report.build_report(tmp_path, data_manifest=data)
  assert not (tmp_path / "research_report.json").exists()


def test_legacy_model_seed_subsets_cannot_be_named_one_shared_research_cohort(tmp_path):
  data = _research_runs(tmp_path, seeds=(42,), selection_seed=42)
  for arm in research_report.ARMS:
    directory = tmp_path / f"{arm}_42"
    config = json.loads((directory / "config.json").read_text())
    config.pop("selection_seed")
    _json(directory / "config.json", config)
    manifest = json.loads((directory / "split_manifest.json").read_text())
    manifest.pop("selection_seed")
    manifest.pop("effective_selection_seed")
    _json(directory / "split_manifest.json", manifest)
  with pytest.raises(ValueError, match="explicit shared selection_seed"):
    research_report.build_report(tmp_path, seeds=(42,), data_manifest=data)


def test_numeric_metrics_count_absent_ids_without_inventing_names():
  matrix = [[0] * 9 for _ in range(9)]
  matrix[0][0] = 10
  scores, ids = research_report._scores(matrix)
  assert scores["macro_f1_all_ids"] == pytest.approx(100 / 9)
  assert scores["mean_iou_present_ids"] == 100
  assert scores["pixel_accuracy"] == 100
  assert ids[1] == {"id": 1, "support": 0, "predicted_pixels": 0,
                    "f1_percent": 0, "iou_percent": None}


def test_single_seed_and_unverified_origin_are_not_real_multiseed_results(tmp_path):
  data = _research_runs(tmp_path, seeds=(42,))
  (tmp_path / "research_spec.json").unlink()
  report = research_report.build_report(tmp_path, seeds=(42,), data_manifest=data)
  assert report["status"] == "artifacts_complete_data_kind_unverified"
  assert report["seed_count"] == 1 and not report["research_pilot_multiseed"]
  assert report["summary"]["cnn_ssm"]["metrics_percent"]["pixel_accuracy"][
      "sample_sd"] is None
  _json(tmp_path / "research_spec.json", {"data_kind": "real_forty"})
  with pytest.raises(ValueError, match="official source"):
    research_report.build_report(tmp_path, seeds=(42,), data_manifest=data)
  assert GCS_ROOT not in str(tmp_path)


def test_optimizer_updates_are_reported_without_requiring_matched_counts(tmp_path):
  data = _research_runs(tmp_path, seeds=(42,))
  for arm, updates in (("cnn_ssm", 9), ("cnn_only", 10)):
    path = tmp_path / f"{arm}_42/training_results.csv"
    with path.open(newline="") as handle:
      rows = list(csv.DictReader(handle))
    for row in rows:
      row.update(train_batches=10, optimizer_updates=updates,
                 skipped_optimizer_updates=10 - updates)
    with path.open("w", newline="") as handle:
      writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
      writer.writeheader()
      writer.writerows(rows)
  report = research_report.build_report(tmp_path, seeds=(42,), data_manifest=data)
  assert report["pairs"][0]["cnn_ssm"]["optimizer_updates"] == {
      "status": "recorded", "train_batches": 30, "optimizer_updates": 27,
      "skipped_optimizer_updates": 3}
  assert report["pairs"][0]["cnn_only"]["optimizer_updates"]["optimizer_updates"] == 30


def test_report_module_and_cli_help_do_not_import_tensorflow(tmp_path):
  script = """import sys
from mamba_forest import research_report
assert 'tensorflow' not in sys.modules
"""
  environment = {**os.environ, "PYTHONPATH": str(
      Path(__file__).resolve().parents[1] / "src")}
  subprocess.run([sys.executable, "-c", script], env=environment, check=True)
  command = [sys.executable, "-m", "mamba_forest.research_report", "--help"]
  result = subprocess.run(command, env=environment, capture_output=True,
                          text=True, check=True)
  assert "--data-manifest" in result.stdout


@pytest.mark.parametrize("filename", [
    "cnn_ssm_42/best.weights.h5", "cnn_only_42/config.json", "selected_data.json",
    "checkpoint_manifest.json", "research_spec.json"])
def test_report_output_never_overwrites_retained_evidence(tmp_path, filename):
  data = _research_runs(tmp_path, seeds=(42,))
  path = tmp_path / filename
  before = path.read_bytes() if path.exists() else None
  with pytest.raises(ValueError):
    research_report.build_report(tmp_path, seeds=(42,), data_manifest=data, output=path)
  assert (path.read_bytes() if path.exists() else None) == before


def test_derived_report_can_be_rebuilt_without_hashing_itself(tmp_path):
  data = _research_runs(tmp_path, seeds=(42,))
  first = research_report.build_report(tmp_path, seeds=(42,), data_manifest=data)
  second = research_report.build_report(tmp_path, seeds=(42,), data_manifest=data)
  assert first == second
  destination = tmp_path / "fresh-summary.json"
  report = research_report.build_report(
      tmp_path, seeds=(42,), data_manifest=data, output=destination)
  assert json.loads(destination.read_text()) == report
  with pytest.raises(ValueError, match="fresh"):
    research_report.build_report(
        tmp_path, seeds=(42,), data_manifest=data, output=destination)


def _real_receipts(root, data):
  staged = json.loads(data.read_text())
  _json(root / "source_identity.json", {
      "commit": "a" * 40, "module_sha256": {"model.py": "b" * 64}})
  _json(root / "runtime_identity.json", {
      "tensorflow": "2.21.0", "keras": "3.13.2", "numpy": "2.1.3",
      "python": "3.13.16", "probe_device": "/device:GPU:0"})
  _json(root / "data_identity.json", {
      "manifest_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
      "config": staged["config"], "files": staged["files"], "data_kind": "real_forty"})


def test_real_pilot_requires_official_data_and_complete_gpu_provenance(tmp_path):
  data = _research_runs(tmp_path, data_kind="real_forty", total_shards=1024,
                        source_root=GCS_ROOT)
  _real_receipts(tmp_path, data)
  report = research_report.build_report(tmp_path, data_manifest=data)
  assert report["status"] == "completed_pilot"
  assert report["provenance_validation"]["status"] == "passed"
  assert report["provenance_validation"]["source_commit"] == "a" * 40
  assert report["coverage"]["published_shards_per_split"] == 1024
  assert not report["coverage"]["full_published_test_split"]


@pytest.mark.parametrize("failure", ["missing", "cpu_probe", "data_hash",
                                     "source_hash", "runtime_version"])
def test_real_pilot_cannot_complete_with_invalid_identity_receipts(tmp_path, failure):
  data = _research_runs(tmp_path, data_kind="real_forty",
                        total_shards=1024, source_root=GCS_ROOT)
  _real_receipts(tmp_path, data)
  if failure == "missing":
    (tmp_path / "source_identity.json").unlink()
  else:
    filename = {"cpu_probe": "runtime_identity.json", "data_hash": "data_identity.json",
                "source_hash": "source_identity.json",
                "runtime_version": "runtime_identity.json"}[failure]
    receipt = json.loads((tmp_path / filename).read_text())
    if failure == "cpu_probe":
      receipt["probe_device"] = "/device:CPU:0"
    elif failure == "data_hash":
      receipt["manifest_sha256"] = "0" * 64
    elif failure == "source_hash":
      receipt["module_sha256"]["model.py"] = "not-a-digest"
    elif failure == "runtime_version":
      receipt["tensorflow"] = "other-version"
    _json(tmp_path / filename, receipt)
  with pytest.raises((ValueError, FileNotFoundError)):
    research_report.build_report(tmp_path, data_manifest=data)
  assert not (tmp_path / "research_report.json").exists()


def test_a_subset_of_planned_seeds_cannot_complete_the_frozen_study(tmp_path):
  data = _research_runs(tmp_path)
  _json(tmp_path / "plan.json", {"config": {"seeds": [42, 43, 44]}})
  with pytest.raises(ValueError, match="frozen research plan"):
    research_report.build_report(tmp_path, seeds=(42,), data_manifest=data)
  report = research_report.build_report(tmp_path, data_manifest=data)
  assert report["seed_count"] == 3


def test_frozen_spec_seeds_and_real_three_seed_minimum_are_enforced(tmp_path):
  data = _research_runs(tmp_path, seeds=(42,), data_kind="real_forty",
                        total_shards=1024, source_root=GCS_ROOT)
  _real_receipts(tmp_path, data)
  with pytest.raises(ValueError, match="at least three seeds"):
    research_report.build_report(tmp_path, seeds=(42,), data_manifest=data)
  _json(tmp_path / "research_spec.json", {
      "data_kind": "real_forty", "config": {"seeds": [42, 43, 44]}})
  with pytest.raises(ValueError, match="frozen research spec"):
    research_report.build_report(tmp_path, seeds=(42,), data_manifest=data)
