"""Paired-run plans are inspectable and execution never reuses stale artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from mamba_forest import experiment


def _arguments(tmp_path, *extra):
  return experiment.parse_args([
      "--run-root", str(tmp_path / "run"), "--data-root", str(tmp_path / "records"),
      "--total-shards", "1", "--train-shards", "1", "--val-shards", "1",
      "--seeds", "42", "--epochs", "2", "--allow-cpu", *extra])


def _json(path, value):
  Path(path).write_text(json.dumps(value), encoding="utf-8")


def _train_artifacts(plan, step, changed_paths=False):
  config = plan["config"]
  directory = Path(config["run_root"]) / f"{step['arm']}_{step['seed']}"
  directory.mkdir(parents=True, exist_ok=True)
  for filename in ("best.weights.h5", "last.weights.h5"):
    (directory / filename).write_bytes(b"checkpoint")
  training = {
      **{key: config[key] for key in (
          "total_shards", "train_shards", "val_shards", "batch_size", "epochs",
          "d_model", "dropout", "d_state_temporal", "d_state_spatial",
          "bottleneck_channels")},
      "gcs_root": config["data_root"], "resolved_precision": config["precision"],
      "precision": config["precision"], "no_mixed_precision": False,
      "seed": step["seed"], "patience": 0, "fixed_train_subset": True,
      "combo_alpha": 1.0 if step["arm"] == "plain_ce" else 0.4,
      "label_smoothing": 0.0 if step["arm"] == "plain_ce" else 0.05,
      "no_class_weights": step["arm"] == "plain_ce",
      "model_variant": (step["arm"] if config["comparison"] == "architecture"
                        else "cnn_ssm"),
  }
  _json(directory / "config.json", training)
  paths = ["train/a" if not changed_paths else "train/different"]
  _json(directory / "split_manifest.json", {
      "training_mode": "fixed", "validation_paths": ["validation/a"],
      "epochs": [{"epoch": index, "training_completed": True, "paths": paths}
                 for index in range(1, config["epochs"] + 1)],
  })
  (directory / "training_results.csv").write_text(
      "epoch,train_loss\n" + "".join(
          f"{index},1.0\n" for index in range(1, config["epochs"] + 1)),
      encoding="utf-8")
  return directory


def _fake_subprocess(monkeypatch, plan, fail=None, changed_pair=False, missing=False):
  calls = []

  def validate_training(directory, arm, seed):
    # Comparator tests cover strict CSV/manifest/checkpoint validation. These
    # orchestration tests isolate process ordering and preservation of outputs.
    return {
        "config": json.loads((directory / "config.json").read_text()),
        "manifest": json.loads((directory / "split_manifest.json").read_text()),
    }

  monkeypatch.setattr(experiment, "validate_training_run", validate_training)

  def run(argv, **kwargs):
    assert isinstance(argv, list) and all(isinstance(value, str) for value in argv)
    assert not kwargs.get("shell", False)
    if kwargs.get("stdout") == subprocess.PIPE:
      return subprocess.CompletedProcess(argv, 0, stdout="snapshot\n")
    step = next(item for item in plan["steps"] if item["argv"] == argv)
    calls.append(step["id"])
    kwargs["stdout"].write(step["id"] + "\n")
    if step["id"] == fail:
      return subprocess.CompletedProcess(argv, 7)
    if step["phase"] == "train":
      directory = _train_artifacts(
          plan, step, changed_paths=changed_pair and step["arm"] in (
              "plain_ce", "cnn_only"))
      if missing:
        (directory / "best.weights.h5").unlink()
    elif step["phase"] == "preflight":
      _json(step["artifact"], {"status": "passed"})
    elif "artifact" in step:
      Path(step["artifact"]).parent.mkdir(parents=True, exist_ok=True)
      _json(step["artifact"], {"num_pixels": 16384})
    return subprocess.CompletedProcess(argv, 0)

  monkeypatch.setattr(experiment.subprocess, "run", run)
  return calls


def test_plan_is_framework_free_and_never_opens_records(tmp_path):
  source = Path(experiment.__file__).resolve().parents[1]
  code = """import sys
from mamba_forest.experiment import parse_args, make_plan
args = parse_args(['--run-root', sys.argv[1], '--data-root', '/does/not/exist'])
plan = make_plan(args)
assert 'tensorflow' not in sys.modules
assert plan['config']['test_shards'] == plan['config']['total_shards'] == 1024
assert plan['config']['precision'] == 'mixed_float16'
"""
  environment = {**os.environ, "PYTHONPATH": str(source)}
  subprocess.run([sys.executable, "-c", code, str(tmp_path / "plan-only")],
                 env=environment, check=True)
  assert not (tmp_path / "plan-only").exists()


def test_plan_commands_fix_pair_conditions_and_separate_final_test(tmp_path):
  plan = experiment.make_plan(_arguments(tmp_path, "--seeds", "42", "43"))
  assert plan["config"]["precision"] == "float32"
  training = [step for step in plan["steps"] if step["phase"] == "train"]
  assert len(training) == 4
  for step in training:
    argv = step["argv"]
    for flag, expected in (("--patience", "0"), ("--epochs", "2"),
                           ("--precision", "float32"), ("--total-shards", "1")):
      assert argv[argv.index(flag) + 1] == expected
    assert "--fixed-train-subset" in argv
    assert ("--no-class-weights" in argv) == (step["arm"] == "plain_ce")
  phases = [step["phase"] for step in plan["steps"]]
  last_training = max(i for i, phase in enumerate(phases) if phase == "train")
  assert last_training < phases.index("evaluate")
  assert plan["steps"][-1]["phase"] == "compare"
  evaluations = [step for step in plan["steps"] if step["phase"] == "evaluate"]
  assert all("--no-config" not in step["argv"] for step in evaluations)
  reduced_args = _arguments(tmp_path, "--total-shards", "4", "--test-shards", "1")
  reduced = experiment.make_plan(reduced_args)
  assert reduced["test_coverage"] == "subset"


def test_architecture_plan_pairs_variants_with_identical_objective(tmp_path):
  plan = experiment.make_plan(_arguments(tmp_path, "--comparison", "architecture"))
  assert plan["config"]["comparison"] == "architecture"
  training = [step for step in plan["steps"] if step["phase"] == "train"]
  assert [step["arm"] for step in training] == ["cnn_ssm", "cnn_only"]
  controls = []
  for step in training:
    argv = step["argv"]
    assert argv[argv.index("--model-variant") + 1] == step["arm"]
    assert argv[argv.index("--combo-alpha") + 1] == "0.4"
    assert argv[argv.index("--label-smoothing") + 1] == "0.05"
    assert "--no-class-weights" not in argv
    ignored = {argv.index("--model-variant") + 1, argv.index("--output-dir") + 1}
    controls.append([value for index, value in enumerate(argv) if index not in ignored])
  assert controls[0] == controls[1]
  comparison = plan["steps"][-1]["argv"]
  assert comparison[comparison.index("--comparison") + 1] == "architecture"
  phases = [step["phase"] for step in plan["steps"]]
  assert max(i for i, value in enumerate(phases) if value == "train") < phases.index(
      "evaluate")


def test_architecture_execution_checks_training_and_preserves_paired_memberships(
      tmp_path, monkeypatch):
  plan = experiment.make_plan(_arguments(tmp_path, "--comparison", "architecture"))
  calls = _fake_subprocess(monkeypatch, plan, changed_pair=True)
  with pytest.raises(ValueError, match="Paired split memberships differ"):
    experiment.execute_plan(plan)
  assert calls[-1] == "train_cnn_only_42"
  assert not any(name.startswith("evaluate") for name in calls)


def test_architecture_plan_rejects_swapped_variant_before_evaluation(tmp_path):
  plan = experiment.make_plan(_arguments(tmp_path, "--comparison", "architecture"))
  step = next(item for item in plan["steps"] if item["id"] == "train_cnn_only_42")
  directory = _train_artifacts(plan, step)
  config = json.loads((directory / "config.json").read_text())
  config["model_variant"] = "cnn_ssm"
  _json(directory / "config.json", config)
  # The shared comparator validator needs full artifacts; isolate the planner
  # contract here and cover actual validation with complete comparator fixtures.
  from unittest.mock import patch
  with patch.object(experiment, "validate_training_run", return_value={
      "config": config, "manifest": json.loads(
          (directory / "split_manifest.json").read_text())}):
    with pytest.raises(ValueError, match="differs from plan"):
      experiment._verify_run(plan["config"], 42, "cnn_only")


def test_plan_only_writes_reviewable_commands_and_identical_plan_reusable(
      tmp_path, monkeypatch):
  plan = experiment.make_plan(_arguments(tmp_path))
  monkeypatch.setattr(experiment.subprocess, "run",
                      lambda *a, **k: pytest.fail("executed"))
  path = experiment.prepare_plan(plan)
  assert json.loads(path.read_text()) == plan
  assert experiment.prepare_plan(plan) == path
  changed = experiment.make_plan(_arguments(tmp_path, "--epochs", "3"))
  with pytest.raises(ValueError, match="Existing plan differs"):
    experiment.prepare_plan(changed)


def test_execute_runs_once_with_logs_wall_times_and_metadata(tmp_path, monkeypatch):
  plan = experiment.make_plan(_arguments(tmp_path))
  calls = _fake_subprocess(monkeypatch, plan)
  state = experiment.execute_plan(plan)
  root = Path(plan["config"]["run_root"])
  assert state["state"] == "completed"
  assert calls == [step["id"] for step in plan["steps"]]
  assert json.loads((root / "execution.json").read_text())["state"] == "completed"
  for step in plan["steps"]:
    assert Path(step["log_path"]).is_file()
    if "wall_seconds_path" in step:
      assert float(Path(step["wall_seconds_path"]).read_text()) > 0
  assert (root / "source_commit.txt").read_text() == "snapshot\n"
  assert (root / "packages.txt").read_text() == "snapshot\n"
  prior_calls = list(calls)
  with pytest.raises(ValueError, match="not fresh"):
    experiment.execute_plan(plan)
  assert calls == prior_calls


@pytest.mark.parametrize("failure", ["runtime_probe", "train_combo_42"])
def test_failure_stops_before_test_and_preserves_returncode(
      tmp_path, monkeypatch, failure):
  plan = experiment.make_plan(_arguments(tmp_path))
  calls = _fake_subprocess(monkeypatch, plan, fail=failure)
  with pytest.raises(RuntimeError, match="exited 7"):
    experiment.execute_plan(plan)
  assert calls[-1] == failure
  assert not any(name.startswith("evaluate") for name in calls)
  root = Path(plan["config"]["run_root"])
  state = json.loads((root / "execution.json").read_text())
  assert state["state"] == "failed"
  failed = next(stage for stage in state["stages"] if stage["id"] == failure)
  assert failed["state"] == "failed" and failed["returncode"] == 7
  assert Path(next(step["log_path"] for step in plan["steps"]
                   if step["id"] == failure)).is_file()


def test_missing_checkpoint_stops_even_after_successful_child(tmp_path, monkeypatch):
  plan = experiment.make_plan(_arguments(tmp_path))
  calls = _fake_subprocess(monkeypatch, plan, missing=True)
  with pytest.raises(ValueError, match="Missing or empty training artifact"):
    experiment.execute_plan(plan)
  assert calls[-1] == "train_combo_42"


def test_changed_paired_membership_rejected_before_any_evaluation(
      tmp_path, monkeypatch):
  plan = experiment.make_plan(_arguments(tmp_path))
  calls = _fake_subprocess(monkeypatch, plan, changed_pair=True)
  with pytest.raises(ValueError, match="Paired split memberships differ"):
    experiment.execute_plan(plan)
  assert calls[-1] == "train_plain_ce_42"
  state = json.loads((tmp_path / "run" / "execution.json").read_text())
  assert state["current_stage"] == "verify_training_artifacts"


def test_shared_training_validator_failure_blocks_evaluation(tmp_path, monkeypatch):
  plan = experiment.make_plan(_arguments(tmp_path))
  calls = _fake_subprocess(monkeypatch, plan)

  def invalid_training(*args):
    raise ValueError("checkpoint selection failed validation")

  monkeypatch.setattr(experiment, "validate_training_run", invalid_training)
  with pytest.raises(ValueError, match="checkpoint selection failed validation"):
    experiment.execute_plan(plan)
  assert calls[-1] == "train_combo_42"
  assert not any(name.startswith("evaluate") for name in calls)


def test_existing_unrelated_files_never_overwritten(tmp_path):
  args = _arguments(tmp_path)
  args.run_root.mkdir()
  retained = args.run_root / "notes.txt"
  retained.write_text("preserve me")
  with pytest.raises(ValueError, match="not fresh"):
    experiment.prepare_plan(experiment.make_plan(args))
  assert retained.read_text() == "preserve me"


@pytest.mark.parametrize("flags", [
    ["--seeds", "42", "42"], ["--epochs", "0"], ["--seeds", "-1"],
    ["--test-shards", "2"], ["--dropout", "nan"], ["--d-state-spatial", "0"],
])
def test_invalid_plan_rejected_before_root_creation(tmp_path, flags):
  with pytest.raises(SystemExit):
    _arguments(tmp_path, *flags)
  assert not (tmp_path / "run").exists()
