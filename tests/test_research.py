"""Research continuation preserves completed work and gates test evaluation."""

from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys

import pytest

from mamba_forest import research, research_report, stage_data


def arguments(tmp_path, *extra):
  return research.parse_args([
      "--run-root", str(tmp_path / "study"), "--data-root", str(tmp_path / "cache"),
      "--data-manifest", str(tmp_path / "study/selected_data.json"),
      "--data-kind", "synthetic", "--allow-cpu", "--total-shards", "2",
      "--train-shards", "1", "--val-shards", "1", "--test-shards", "1",
      "--epochs", "2", *extra])


def staged(tmp_path):
  source = tmp_path / "original"
  source.mkdir()
  for record in stage_data.selected_files(str(source), total_shards=2,
                                          train_shards=1, val_shards=1, test_shards=1):
    (source / record["filename"]).write_bytes(record["filename"].encode())
  manifest = tmp_path / "study/selected_data.json"
  stage_data.stage_dataset(source, tmp_path / "cache", manifest,
                           total_shards=2, train_shards=1, val_shards=1, test_shards=1)


def fake_workers(monkeypatch, plan, fail=None):
  root = Path(plan["config"]["run_root"])
  calls, failed = [], False

  def start_session(root, config, args):
    data = research.validate_data(args.data_manifest, config, args.data_kind)
    for name, value in (("source_identity.json", {"source": "same"}),
                        ("runtime_identity.json", {"runtime": "same"}),
                        ("data_identity.json", data)):
      research.freeze(root / name, value, name)
    return "test-session"

  def outputs(root, step, plan):
    if step["phase"] == "train":
      directory = root / f"{step['arm']}_{step['seed']}"
      paths = [directory / name for name in (
          "best.weights.h5", "last.weights.h5", "checkpoint_selection.json")]
    else:
      paths = [Path(step["artifact"])]
    return [research.artifact_record(root, path) for path in paths]

  def command(argv, log):
    nonlocal failed
    step = next(step for step in plan["steps"] if step["argv"] == argv)
    calls.append(step["id"])
    log.parent.mkdir(exist_ok=True)
    log.write_text(step["id"])
    if step["phase"] == "train":
      directory = root / f"{step['arm']}_{step['seed']}"
      directory.mkdir()
      (directory / "best.weights.h5").write_bytes(step["id"].encode())
      if step["id"] == fail and not failed:
        failed = True
        raise RuntimeError("simulated interruption")
      (directory / "last.weights.h5").write_bytes(b"last weights")
      research.save(directory / "checkpoint_selection.json", {"epoch": 2})
    else:
      if step["phase"] == "evaluate":
        state = research.read(root / "research_state.json")
        training = [value["state"] for key, value in state["stages"].items()
                    if key.startswith("train_")]
        assert training == ["completed"] * 6
      research.save(step["artifact"], {"status": "passed"})

  monkeypatch.setattr(research, "start_session", start_session)
  monkeypatch.setattr(research, "verify_session_identity", lambda *args: None)
  monkeypatch.setattr(research, "stage_outputs", outputs)
  monkeypatch.setattr(research, "checked_command", command)
  monkeypatch.setattr(
      research.experiment, "_verify_run",
      lambda *args: (["same-training-shard"], ["same-validation-shard"]))
  monkeypatch.setattr(research_report, "build_report", lambda *args, **kwargs: {})
  return calls


def test_plan_is_framework_free_and_counterbalances_one_shared_cohort(tmp_path):
  code = """import sys
from mamba_forest import research
args=research.parse_args(['--run-root',sys.argv[1],'--data-root','/missing',
                        '--data-manifest','/missing.json'])
plan=research.make_plan(args)
assert 'tensorflow' not in sys.modules
assert plan['config']['selection_seed']==2026
assert len([s for s in plan['steps'] if s['phase']=='train'])==6
"""
  subprocess.run([sys.executable, "-c", code, str(tmp_path / "not-created")],
                 check=True)
  assert not (tmp_path / "not-created").exists()
  plan = research.make_plan(arguments(tmp_path))
  phases = [step["phase"] for step in plan["steps"]]
  assert max(i for i, value in enumerate(phases) if value == "train") < phases.index(
      "evaluate")
  order = [step["id"] for step in plan["steps"] if step["phase"] == "train"]
  assert order == ["train_cnn_ssm_42", "train_cnn_only_42", "train_cnn_only_43",
                   "train_cnn_ssm_43", "train_cnn_ssm_44", "train_cnn_only_44"]


def test_resume_skips_verified_stages_and_preserves_failed_attempt(
        tmp_path, monkeypatch):
  staged(tmp_path)
  args = arguments(tmp_path, "--max-train-runs", "1")
  plan = research.make_plan(args)
  calls = fake_workers(monkeypatch, plan, fail="train_cnn_ssm_43")
  assert research.execute(plan, args)["state"] == "training_pending"
  root = args.run_root
  selected = root / "cnn_ssm_42/best.weights.h5"
  original = selected.read_bytes(), selected.stat().st_mtime_ns
  args.resume, args.max_train_runs = True, None
  with pytest.raises(RuntimeError, match="simulated"):
    research.execute(plan, args)
  assert not any(value.startswith("evaluate") for value in calls)
  partial = (root / "cnn_ssm_43/best.weights.h5").read_bytes()
  result = research.execute(plan, args)
  assert result["state"] == "completed" and "error" not in result
  assert calls.count("train_cnn_ssm_42") == 1
  assert calls.count("train_cnn_ssm_43") == 2
  assert len([value for value in calls if value.startswith("evaluate")]) == 6
  assert (selected.read_bytes(), selected.stat().st_mtime_ns) == original
  saved = list((root / "attempts").rglob("best.weights.h5"))
  assert len(saved) == 1 and saved[0].read_bytes() == partial
  inventory = research.read(root / "checkpoint_inventory.json")
  assert inventory["completed_training_runs"] == 6
  assert inventory["required_training_runs"] == 6
  assert not (root / ".research.lock").exists()


def test_modified_completed_checkpoint_is_not_retrained_or_overwritten(
        tmp_path, monkeypatch):
  staged(tmp_path)
  args = arguments(tmp_path, "--max-train-runs", "1")
  plan = research.make_plan(args)
  calls = fake_workers(monkeypatch, plan)
  research.execute(plan, args)
  checkpoint = args.run_root / "cnn_ssm_42/best.weights.h5"
  checkpoint.write_bytes(b"modified checkpoint")
  previous = list(calls)
  args.resume = True
  with pytest.raises(ValueError, match="artifacts changed"):
    research.execute(plan, args)
  assert calls == previous and checkpoint.read_bytes() == b"modified checkpoint"


@pytest.mark.parametrize("missing", ["source_identity.json", "research_state.json"])
def test_lost_identity_or_execution_ledger_is_not_reconstructed(
        tmp_path, monkeypatch, missing):
  staged(tmp_path)
  args = arguments(tmp_path, "--max-train-runs", "1")
  plan = research.make_plan(args)
  calls = fake_workers(monkeypatch, plan)
  research.execute(plan, args)
  previous = list(calls)
  (args.run_root / missing).unlink()
  args.resume = True
  with pytest.raises(ValueError, match="missing"):
    research.execute(plan, args)
  assert calls == previous
  assert not (args.run_root / missing).exists()


def test_unknown_stage_state_does_not_restart_training(tmp_path, monkeypatch):
  staged(tmp_path)
  args = arguments(tmp_path, "--max-train-runs", "0")
  plan = research.make_plan(args)
  calls = fake_workers(monkeypatch, plan)
  research.execute(plan, args)
  path = args.run_root / "research_state.json"
  state = research.read(path)
  state["stages"]["train_cnn_ssm_42"]["state"] = "unexpected"
  research.save(path, state)
  args.resume = True
  with pytest.raises(ValueError, match="Unknown saved stage"):
    research.execute(plan, args)
  assert not any(value.startswith("train_") for value in calls)


def test_planning_cannot_replace_a_manifest_or_mutate_an_active_study(tmp_path):
  args = arguments(tmp_path)
  root = args.run_root
  root.mkdir()
  args.data_manifest = root / "plan.json"
  args.data_manifest.write_text('{"staged_input":"preserve"}')
  with pytest.raises(ValueError, match="reserved"):
    research.prepare(research.make_plan(args), args)
  assert args.data_manifest.read_text() == '{"staged_input":"preserve"}'
  args.data_manifest.unlink()
  args.data_manifest = root / "selected_data.json"
  plan = research.make_plan(args)
  research.prepare(plan, args)
  original = (root / "research_spec.json").read_bytes()
  args.epochs += 1
  with research.study_lock(root):
    with pytest.raises(ValueError, match="still running"):
      research.prepare(research.make_plan(args), args)
  assert (root / "research_spec.json").read_bytes() == original


def test_recovery_does_not_replace_a_newer_lock(tmp_path, monkeypatch):
  path = tmp_path / ".research.lock"
  path.write_text(json.dumps({"host": "previous-host", "pid": 1, "token": "old"}))
  replacement = {"host": "new-host", "pid": 2, "token": "new"}

  @contextmanager
  def concurrent_recovery(root):
    research.save(path, replacement)
    yield

  monkeypatch.setattr(research, "recovery_guard", concurrent_recovery)
  with pytest.raises(ValueError, match="changed during recovery"):
    with research.study_lock(tmp_path, recover=True):
      pytest.fail("must not acquire a competing lock")
  assert research.read(path) == replacement


def test_recovering_an_interrupted_lock_keeps_its_raw_bytes(tmp_path):
  path = tmp_path / ".research.lock"
  path.write_bytes(b"incomplete JSON")
  with research.study_lock(tmp_path, recover=True):
    assert path.exists()
  previous = list((tmp_path / "attempts").glob("*.txt"))
  assert len(previous) == 1 and previous[0].read_bytes() == b"incomplete JSON"


def test_data_and_source_identity_changes_are_rejected(tmp_path, monkeypatch):
  staged(tmp_path)
  args = arguments(tmp_path)
  plan = research.make_plan(args)
  research.validate_data(args.data_manifest, plan["config"], "synthetic")
  manifest = research.read(args.data_manifest)
  (Path(args.data_root) / manifest["files"][0]["filename"]).write_bytes(b"changed")
  with pytest.raises(ValueError, match="bytes changed"):
    research.validate_data(args.data_manifest, plan["config"], "synthetic")
  source = {"commit": "a" * 40, "module_sha256": {"train.py": "b" * 64}}
  research.save(tmp_path / "source_identity.json", source)
  monkeypatch.setattr(research, "source_snapshot",
                      lambda: {**source, "commit": "c" * 40})
  with pytest.raises(ValueError, match="Source changed"):
    research.verify_session_identity(tmp_path, "synthetic")


@pytest.mark.parametrize("flags", [
    ["--seeds", "42"], ["--allow-cpu"], ["--total-shards", "1"],
    ["--max-train-runs", "-1"],
])
def test_real_pilot_rejects_inadequate_protocol_before_io(flags):
  with pytest.raises(SystemExit):
    research.parse_args(["--run-root", "unused", "--data-root", "missing",
                         "--data-manifest", "missing.json", *flags])
