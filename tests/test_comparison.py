"""Offline paired-comparison integrity checks using local artifact fixtures."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mamba_forest import compare  # noqa: E402
from mamba_forest.constants import CLASS_NAMES, NUM_CLASSES  # noqa: E402


def _write_json(path, value):
  path.write_text(json.dumps(value), encoding="utf-8")


def _change_json(path, change):
  value = json.loads(path.read_text(encoding="utf-8"))
  change(value)
  _write_json(path, value)


def _write_run(root, variant, seed, correct=7,
               best_scores=(0.1, 0.40000001, 0.40000002),
               selection_seed=None, updates_per_epoch=None):
  directory = root / f"{variant}_{seed}"
  directory.mkdir(parents=True)
  config = {
      "output_dir": str(directory), "seed": seed, "gcs_root": "local-records",
      "total_shards": 4, "train_shards": 2, "val_shards": 1, "batch_size": 2,
      "epochs": len(best_scores), "fixed_train_subset": True, "learning_rate": 2e-4,
      "lr_alpha": 0.1, "weight_decay": 0.005, "d_model": 4, "dropout": 0.2,
      "d_state_temporal": 2, "d_state_spatial": 2, "bottleneck_channels": 8,
      "grad_clip": 1.0, "patience": 0, "precision": "auto",
      "resolved_precision": "float32", "no_mixed_precision": False,
      "tensorflow_version": "2.21.0",
      "combo_alpha": 1.0 if variant == "plain_ce" else 0.4,
      "label_smoothing": 0.0 if variant == "plain_ce" else 0.05,
      "no_class_weights": variant == "plain_ce",
  }
  if variant in ("cnn_ssm", "cnn_only"):
    config["model_variant"] = variant
  if selection_seed is not None:
    config["selection_seed"] = selection_seed
  _write_json(directory / "config.json", config)

  def paths(split):
    return [f"local-records/forty_v1-{split}.tfrecord-{index:05d}-of-00004"
            for index in range(4)]

  training_paths = random.Random(
      seed if selection_seed is None else selection_seed).sample(paths("train"), 2)
  manifest = {
      "schema_version": 1, "selection_protocol": "python-random-seed-plus-epoch-v1",
      "gcs_root": "local-records", "total_shards_per_split": 4, "seed": seed,
      "training_mode": "fixed", "validation_paths": paths("validation")[:1],
      "completed_unique_train_shards": 2,
      "epochs": [{"epoch": epoch, "shard_epoch": 0, "training_completed": True,
                  "paths": training_paths} for epoch in range(1, len(best_scores) + 1)],
  }
  if selection_seed is not None:
    manifest.update(selection_seed=selection_seed,
                    effective_selection_seed=selection_seed)
  if updates_per_epoch is not None:
    counts = {"train_batches": 10, "optimizer_updates": updates_per_epoch,
              "skipped_optimizer_updates": 10 - updates_per_epoch}
    for entry in manifest["epochs"]:
      entry.update(counts)
  _write_json(directory / "split_manifest.json", manifest)
  rows = []
  for epoch, score in enumerate(best_scores, 1):
    rows.append({
        "epoch": epoch, "train_loss": 0.8, "val_accuracy": 0.5, "val_mean_iou": 0.3,
        "overall_f1": score, "forests_f1": score,
        **{"f1_" + name.lower().replace(" ", "_"): score for name in CLASS_NAMES},
        "train_seconds": epoch, "validation_seconds": 0.5,
        "train_examples": 20, "train_examples_per_second": 20 / epoch,
    })
    if updates_per_epoch is not None:
      rows[-1].update(counts)
  with (directory / "training_results.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  best = max(range(len(best_scores)), key=lambda index: best_scores[index])
  weights = f"local checkpoint {variant} seed {seed}".encode()
  (directory / "best.weights.h5").write_bytes(weights)
  _write_json(directory / "checkpoint_selection.json", {
      "schema_version": 1, "criterion": "validation_overall_f1", "mode": "max",
      "tie_break": "first", "epoch": best + 1, "score": best_scores[best],
      "weights": "best.weights.h5",
      "weights_sha256": hashlib.sha256(weights).hexdigest(),
  })
  matrix = [[0] * NUM_CLASSES for _ in CLASS_NAMES]
  for index in range(NUM_CLASSES):
    matrix[index][index] = correct
    matrix[index][(index + 1) % NUM_CLASSES] = 10 - correct
  score = correct / 10
  metrics = {
      "weights": str(directory / "best.weights.h5"),
      "weights_sha256": hashlib.sha256(weights).hexdigest(),
      "training_config_path": str(directory / "config.json"),
      "gcs_root": "local-records", "split": "test", "test_shards": 1,
      "total_shards_per_split": 4, "shard_paths": paths("test")[:1], "batch_size": 2,
      "model_config": {"name": f"model_{variant}", "num_classes": NUM_CLASSES,
                       "d_model": 4, "dropout_rate": 0.2, "d_state_temporal": 2,
                       "d_state_spatial": 2, "bottleneck_channels": 8},
      "runtime": {"tensorflow_version": "2.21.0", "precision_policy": "float32"},
      "confusion_axes": {"rows": "true", "columns": "predicted"},
      "confusion_matrix": matrix, "num_pixels": 90,
      "class_support": dict.fromkeys(CLASS_NAMES, 10),
      "metric_conventions": {
          "overall_f1": "macro mean over all classes; absent classes have F1 0",
          "mean_iou": "mean over classes with nonzero true or predicted support",
      },
      "pixel_accuracy": score, "mean_iou": correct / (20 - correct),
      "per_class_f1": dict.fromkeys(CLASS_NAMES, score),
      **dict.fromkeys(compare.SCORE_KEYS, score),
  }
  if variant in ("cnn_ssm", "cnn_only"):
    metrics["model_config"]["model_variant"] = variant
    metrics["parameters"] = 120 if variant == "cnn_ssm" else 100
  (directory / "test").mkdir()
  _write_json(directory / "test" / "test_metrics.json", metrics)
  return directory


def _pair(root, seed=42, combo=7, plain=6, **kwargs):
  return (_write_run(root, "combo", seed, combo, **kwargs),
          _write_run(root, "plain_ce", seed, plain, **kwargs))


def _architecture_pair(root, seed=42, cnn_ssm=7, cnn_only=6):
  return (_write_run(root, "cnn_ssm", seed, cnn_ssm),
          _write_run(root, "cnn_only", seed, cnn_only))


def test_architecture_scores_include_miou_and_unmatched_parameter_budgets(tmp_path):
  _architecture_pair(tmp_path, 42)
  _architecture_pair(tmp_path, 43, cnn_ssm=8, cnn_only=6)
  report = compare.summarize_runs(tmp_path, [42, 43], comparison="architecture")
  assert report["comparison"] == "architecture"
  assert report["delta_direction"] == "cnn_ssm_minus_cnn_only"
  assert report["summary"]["cnn_ssm"]["parameter_count"] == 120
  assert report["summary"]["cnn_only"]["parameter_count"] == 100
  assert report["pairs"][0]["cnn_ssm"]["scores_percent"]["mean_iou"] == pytest.approx(
      100 * 7 / 13)
  assert report["summary"]["paired_delta_percentage_points"]["overall_f1"][
      "mean"] == pytest.approx(15)
  assert "not matched" in report["architecture_scope"]


@pytest.mark.parametrize("filename,change,message", [
    ("config.json", lambda value: value.update(combo_alpha=1.0), "objective"),
    ("config.json", lambda value: value.update(learning_rate=0.01), "config mismatch"),
    ("config.json", lambda value: value.update(model_variant="cnn_ssm"),
     "model variant"),
    ("config.json", lambda value: value.pop("model_variant"), "model variant"),
    ("test/test_metrics.json",
     lambda value: value["model_config"].update(model_variant="cnn_ssm"),
     "model variant differs"),
    ("test/test_metrics.json", lambda value: value.update(parameters=-1),
     "parameter_count"),
])
def test_architecture_comparison_rejects_changed_controls_or_variants(
      tmp_path, filename, change, message):
  _, baseline = _architecture_pair(tmp_path)
  _change_json(baseline / filename, change)
  with pytest.raises(ValueError, match=message):
    compare.summarize_runs(tmp_path, [42], comparison="architecture")


def test_objective_comparison_rejects_architecture_change_and_accepts_legacy_default(
      tmp_path):
  combo, plain = _pair(tmp_path)
  _change_json(combo / "config.json",
               lambda value: value.update(model_variant="cnn_ssm"))
  _change_json(combo / "test" / "test_metrics.json", lambda value: value[
      "model_config"].update(model_variant="cnn_ssm"))
  assert compare.summarize_runs(tmp_path, [42])["comparison"] == "objective"
  _change_json(plain / "config.json",
               lambda value: value.update(model_variant="cnn_only"))
  with pytest.raises(ValueError, match="model variant"):
    compare.summarize_runs(tmp_path, [42])


def test_architecture_comparison_rejects_parameter_count_changes_across_seeds(tmp_path):
  _architecture_pair(tmp_path, 42)
  _, baseline = _architecture_pair(tmp_path, 43)
  _change_json(baseline / "test" / "test_metrics.json",
               lambda value: value.update(parameters=101))
  with pytest.raises(ValueError, match="parameter counts differ across seeds"):
    compare.summarize_runs(tmp_path, [42, 43], comparison="architecture")


def test_single_seed_scores_deltas_selection_and_timing_are_auditable(tmp_path):
  _pair(tmp_path)
  report = compare.summarize_runs(tmp_path, [42])
  pair = report["pairs"][0]
  assert set(pair["combo"]["scores_percent"]) == set(compare.SCORE_KEYS)
  assert pair["combo"]["scores_percent"]["overall_f1"] == 70
  assert pair["plain_ce"]["scores_percent"]["overall_f1"] == 60
  assert pair["paired_delta_percentage_points"]["overall_f1"] == pytest.approx(10)
  assert pair["combo"]["best_validation_epoch"] == 3
  assert pair["combo"]["train_seconds"] == 6
  assert pair["combo"]["validation_seconds"] == 1.5
  assert pair["combo"]["train_examples"] == 60
  assert pair["combo"]["train_examples_per_second"] == 10
  assert pair["combo"]["epoch_timing"]["first_epoch"]["train_seconds"] == 1
  assert pair["combo"]["epoch_timing"]["later_epochs"]["train_seconds"] == 5
  assert pair["combo"]["train_wall_seconds"] is None
  assert pair["combo"]["optimizer_updates"] is None
  assert report["selection_scope"] == {
      "selection_seed": None, "training_membership": "paired_within_each_seed"}
  assert report["summary"]["combo"]["scores_percent"]["overall_f1"] == {
      "mean": 70, "sample_sd": None}
  assert report["summary"]["combo"]["train_examples_per_second"] == 10
  assert report["test_coverage"] == {
      "shards": 1, "total_shards": 4, "full_split": False}


def test_shared_cohort_and_skip_counts_do_not_force_identical_updates(tmp_path):
  memberships = []
  for seed in (42, 43, 44):
    for variant, updates in (("cnn_ssm", 8), ("cnn_only", 10)):
      directory = _write_run(tmp_path, variant, seed, selection_seed=2026,
                             updates_per_epoch=updates)
      memberships.append(json.loads(
          (directory / "split_manifest.json").read_text())["epochs"][0]["paths"])
  assert all(paths == memberships[0] for paths in memberships)
  report = compare.summarize_runs(tmp_path, [42, 43, 44], "architecture")
  assert report["selection_scope"] == {
      "selection_seed": 2026, "training_membership": "shared_across_seeds"}
  assert report["summary"]["cnn_ssm"]["optimizer_updates"]["mean"] == 24
  assert report["summary"]["cnn_only"]["optimizer_updates"]["mean"] == 30
  assert report["summary"]["cnn_ssm"]["skipped_optimizer_updates"]["mean"] == 6
  wrong = tmp_path / "cnn_only_42/split_manifest.json"
  _change_json(wrong, lambda value: value.update(selection_seed=7))
  with pytest.raises(ValueError, match="selection seed"):
    compare.summarize_runs(tmp_path, [42, 43, 44], "architecture")


@pytest.mark.parametrize("change,message", [
    ({"optimizer_updates": 9, "skipped_optimizer_updates": 0}, "batch budget"),
    ({"train_batches": 11, "optimizer_updates": 11,
      "skipped_optimizer_updates": 0}, "batch count"),
    ({"optimizer_updates": 0, "skipped_optimizer_updates": 10}, "below 1"),
])
def test_comparator_rejects_inconsistent_update_progress(tmp_path, change, message):
  directory = _write_run(tmp_path, "combo", 42, updates_per_epoch=10)
  _write_run(tmp_path, "plain_ce", 42, updates_per_epoch=10)
  path = directory / "training_results.csv"
  with path.open() as handle:
    rows = list(csv.DictReader(handle))
  rows[0].update(change)
  with path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  with pytest.raises(ValueError, match=message):
    compare.summarize_runs(tmp_path, [42])


def test_multiseed_sample_sd_and_percentage_point_pairing(tmp_path):
  _pair(tmp_path, 42, combo=7, plain=6)
  _pair(tmp_path, 43, combo=9, plain=5)
  report = compare.summarize_runs(tmp_path, [42, 43])
  overall = report["summary"]["combo"]["scores_percent"]["overall_f1"]
  delta = report["summary"]["paired_delta_percentage_points"]["overall_f1"]
  assert overall["mean"] == 80
  assert overall["sample_sd"] == pytest.approx(math.sqrt(200))
  assert delta["mean"] == pytest.approx(25)
  assert delta["sample_sd"] == pytest.approx(math.sqrt(450))


def test_checkpoint_ties_use_first_maximum_and_wall_times_are_optional(tmp_path):
  combo, plain = _pair(tmp_path, best_scores=(0.2, 0.4, 0.4))
  for directory in (combo, plain):
    (directory / "train_wall_seconds.txt").write_text("8.5\n")
    (directory / "test_wall_seconds.txt").write_text("1.25\n")
  report = compare.summarize_runs(tmp_path, [42])
  assert report["pairs"][0]["combo"]["best_validation_epoch"] == 2
  assert report["summary"]["combo"]["train_wall_seconds"] == {
      "mean": 8.5, "sample_sd": None}
  _change_json(combo / "checkpoint_selection.json", lambda value: value.update(epoch=3))
  with pytest.raises(ValueError, match="selection metadata"):
    compare.summarize_runs(tmp_path, [42])


@pytest.mark.parametrize("filename,change,message", [
    ("config.json", lambda value: value.update(learning_rate=0.01), "config mismatch"),
    ("config.json", lambda value: value.update(combo_alpha=0.5), "objective"),
    ("config.json", lambda value: value.update(fixed_train_subset=False), "fixed"),
    ("config.json", lambda value: value.update(epochs=4), "complete"),
    ("split_manifest.json",
     lambda value: value["epochs"][1].update(training_completed=False),
     "incomplete"),
    ("split_manifest.json",
     lambda value: value["epochs"][1].update(paths=["different"]),
     "fixed subset"),
    ("split_manifest.json", lambda value: value.update(validation_paths=["different"]),
     "validation paths"),
    ("checkpoint_selection.json", lambda value: value.update(score=0.99),
     "selection score"),
    ("test/test_metrics.json", lambda value: value.update(num_pixels=91), "total"),
    ("test/test_metrics.json", lambda value: value.update(overall_f1=0.99),
     "overall_f1"),
    ("test/test_metrics.json", lambda value: value["per_class_f1"].update(Water=0.99),
     "F1 Water"),
    ("test/test_metrics.json", lambda value: value["class_support"].update(Water=11),
     "class_support"),
    ("test/test_metrics.json", lambda value: value.update(shard_paths=["different"]),
     "test shard paths"),
    ("test/test_metrics.json",
     lambda value: value["runtime"].update(precision_policy="mixed_float16"),
     "test protocol"),
    ("test/test_metrics.json", lambda value: value["model_config"].update(d_model=8),
     "model d_model"),
])
def test_unmatched_or_incomplete_artifacts_fail_closed(
      tmp_path, filename, change, message):
  combo, _ = _pair(tmp_path)
  _change_json(combo / filename, change)
  with pytest.raises(ValueError, match=message):
    compare.summarize_runs(tmp_path, [42])


@pytest.mark.parametrize("bad_count", [-1, 0.5, True, float("nan")])
def test_invalid_confusion_counts_are_rejected(tmp_path, bad_count):
  combo, _ = _pair(tmp_path)
  _change_json(combo / "test" / "test_metrics.json",
               lambda value: value["confusion_matrix"][0].__setitem__(0, bad_count))
  with pytest.raises(ValueError):
    compare.summarize_runs(tmp_path, [42])


def test_changed_checkpoint_bytes_cannot_reuse_old_test_metrics(tmp_path):
  combo, _ = _pair(tmp_path)
  (combo / "best.weights.h5").write_bytes(b"checkpoint changed after evaluation")
  with pytest.raises(ValueError, match="SHA256"):
    compare.summarize_runs(tmp_path, [42])


def test_replaced_checkpoint_with_fresh_test_hash_still_fails_selected_identity(
      tmp_path):
  combo, _ = _pair(tmp_path)
  replacement = b"different checkpoint evaluated after training"
  (combo / "best.weights.h5").write_bytes(replacement)
  _change_json(combo / "test" / "test_metrics.json", lambda value: value.update(
      weights_sha256=hashlib.sha256(replacement).hexdigest()))
  with pytest.raises(ValueError, match="training-selected checkpoint SHA256"):
    compare.summarize_runs(tmp_path, [42])


def test_missing_controls_and_inconsistent_validation_metrics_fail(tmp_path):
  combo, plain = _pair(tmp_path)
  for directory in (combo, plain):
    _change_json(directory / "config.json", lambda value: value.pop("learning_rate"))
  with pytest.raises(ValueError, match="missing training condition"):
    compare.summarize_runs(tmp_path, [42])
  for directory in (combo, plain):
    _change_json(directory / "config.json",
                 lambda value: value.update(learning_rate=2e-4))
  path = combo / "training_results.csv"
  path.write_text(path.read_text().replace("0.40000001", "0.45", 1))
  with pytest.raises(ValueError, match="differs from per-class F1"):
    compare.summarize_runs(tmp_path, [42])


def test_training_validation_does_not_open_test_artifacts(tmp_path):
  combo, _ = _pair(tmp_path)
  (combo / "test" / "test_metrics.json").unlink()
  training = compare.validate_training_run(combo, "combo", 42)
  assert training["best"]["epoch"] == 3
  assert len(training["rows"]) == 3


def test_training_examples_and_nonfinite_csv_are_rejected(tmp_path):
  combo, _ = _pair(tmp_path)
  path = combo / "training_results.csv"
  text = path.read_text()
  path.write_text(text.replace(",20,", ",21,"))
  with pytest.raises(ValueError, match="example budgets"):
    compare.summarize_runs(tmp_path, [42])
  path.write_text(text.replace("0.8", "nan"))
  with pytest.raises(ValueError, match="finite"):
    compare.summarize_runs(tmp_path, [42])


def test_missing_artifacts_and_duplicate_seeds_are_rejected(tmp_path):
  combo, _ = _pair(tmp_path)
  with pytest.raises(ValueError, match="unique seed"):
    compare.summarize_runs(tmp_path, [42, 42])
  (combo / "checkpoint_selection.json").unlink()
  with pytest.raises(FileNotFoundError):
    compare.summarize_runs(tmp_path, [42])


def test_cli_writes_default_output_only_for_valid_pairs(tmp_path, monkeypatch):
  combo, _ = _pair(tmp_path)
  monkeypatch.setattr(sys, "argv", [
      "compare", "--run-root", str(tmp_path), "--seeds", "42"])
  compare.main()
  output = tmp_path / "comparison.json"
  report = json.loads(output.read_text())
  assert report["seeds"] == [42]
  output.unlink()
  (combo / "best.weights.h5").unlink()
  with pytest.raises(SystemExit) as error:
    compare.main()
  assert error.value.code == 2
  assert not output.exists()


def test_comparison_module_import_never_loads_tensorflow():
  source = ("import sys; from mamba_forest import compare; "
            "assert not any(name == 'tensorflow' or name.startswith('tensorflow.') "
            "for name in sys.modules)")
  environment = dict(os.environ, PYTHONPATH=os.path.abspath("src"))
  subprocess.run([sys.executable, "-c", source], env=environment, check=True)
