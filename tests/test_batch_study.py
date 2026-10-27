"""Batch studies reject mixed workloads and preserve incomplete OOM cases."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/study_batches.py"
SPEC = importlib.util.spec_from_file_location("batch_study", SCRIPT)
STUDY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STUDY)


def fixture(source="/source"):
  return {"weights_sha256": "weights", "inputs_sha256": "inputs",
          "model_config": {"num_classes": 9},
          "input_shapes": {"s2": [1, 4, 8, 8, 10], "elevation": [1, 8, 8, 3],
                           "climate": [1, 4, 14], "labels": [1, 8, 8]},
          "source": {"commit": "same", "worktree_status": "",
                     "module_sha256": {"model.py": "same"},
                     "package_directory": source + "/src/mamba_forest"},
          "runtime": {"requested_device": "/CPU:0"}}


def measurement(entry, shared):
  batch = entry["batch_size"]
  samples = [0.02, 0.04] if batch == 1 else [0.03, 0.05]
  return {**shared, "status": "passed", "phase": "measure",
          "mode": entry["mode"], "precision": entry["precision"],
          "batch_size": batch,
          "input_shapes": {key: [batch, *shape[1:]]
                           for key, shape in shared["input_shapes"].items()},
          "batch_replication": {"kind": "repeat_single_fixture_tile",
                                "repeat_count": batch, "fixture_batch_size": 1},
          "steps": 2, "warmup_steps": 1, "parameters": 123,
          "latencies_seconds": samples, "median_seconds": sum(samples) / 2,
          "tiles_per_second": batch * 2 / sum(samples),
          "optimizer_updates": 2 if entry["mode"] == "training" else None,
          "forward_agreement": {"passed": True},
          "memory": {"status": "not_applicable_on_cpu"}}


def saved_study(root):
  shared = fixture()
  plan = {"batch_sizes": [1, 2], "rounds": 2, "steps": 2, "warmup": 1,
          "profile_steps": 1, "measurements": []}
  for mode in ("inference", "training"):
    for batch in (1, 2):
      for index in range(2):
        entry = {"phase": "measure", "mode": mode, "precision": "float32",
                 "batch_size": batch, "round": index,
                 "file": f"{mode}-{batch}-{index}.json"}
        plan["measurements"].append(entry)
        (root / entry["file"]).write_text(json.dumps(measurement(entry, shared)))
  return shared, plan


def test_batch_throughput_uses_actual_tiles_and_preserves_rounds(tmp_path):
  shared, plan = saved_study(tmp_path)
  result = STUDY.summarize(tmp_path, plan, shared)
  baseline, larger = result["summary"]["inference"]["batches"]
  assert baseline["samples"] == larger["samples"] == 4
  assert baseline["tiles_per_second"] == pytest.approx(4 / 0.12)
  assert larger["tiles_per_second"] == pytest.approx(8 / 0.16)
  assert larger["throughput_ratio_to_batch_one"] == pytest.approx(1.5)
  assert larger["median_seconds"] == pytest.approx(0.04)
  assert larger["amortized_median_seconds_per_tile"] == pytest.approx(0.02)
  assert len(larger["round_medians_seconds"]) == 2
  assert larger["median_allocator_peak_bytes"] is None


@pytest.mark.parametrize("key,value", [
    ("weights_sha256", "other"), ("inputs_sha256", "other"),
    ("batch_size", 8), ("batch_replication", {"kind": "none"}),
    ("source", {"commit": "other"}),
    ("runtime", {"requested_device": "/GPU:0"}),
    ("input_shapes", {"s2": [8, 4, 8, 8, 10]}),
    ("latencies_seconds", [0.01]), ("tiles_per_second", 999),
    ("optimizer_updates", 0), ("forward_agreement", {"passed": False}),
])
def test_rejects_inconsistent_batch_report(tmp_path, key, value):
  shared, plan = saved_study(tmp_path)
  entry = next(x for x in plan["measurements"]
               if x["mode"] == "training" and x["batch_size"] == 2)
  path = tmp_path / entry["file"]
  report = json.loads(path.read_text())
  report[key] = value
  path.write_text(json.dumps(report))
  with pytest.raises(ValueError):
    STUDY.summarize(tmp_path, plan, shared)


def test_partial_oom_rounds_are_not_used_for_throughput(tmp_path):
  shared, plan = saved_study(tmp_path)
  failure = tmp_path / "training-2-1.json"
  failure.write_text(json.dumps({"status": "failed",
                                 "error_type": "ResourceExhaustedError"}))
  result = STUDY.summarize(tmp_path, plan, shared)
  case = result["summary"]["training"]["batches"][1]
  assert case == {"batch_size": 2, "status": "out_of_memory", "completed_rounds": 1}
  assert result["summary"]["training"]["highest_observed_throughput_batch"] == 1
  failure.unlink()
  with pytest.raises(ValueError, match="Incomplete"):
    STUDY.summarize(tmp_path, plan, shared)


@pytest.mark.parametrize("arguments", [
    ["--batch-sizes", "2", "4"], ["--batch-sizes", "1", "1"],
    ["--batch-sizes", "1", "0"], ["--steps", "0"],
    ["--device", "/GPU:-1"], ["--tile-size", "7"],
])
def test_invalid_study_plan_fails_before_work(arguments):
  with pytest.raises(SystemExit):
    STUDY.parse_args(["--run-root", "unused", *arguments])


@pytest.mark.parametrize("worker_failure", [None, "oom", "other", "profile"])
def test_driver_isolates_jobs_orders_rounds_and_keeps_failures(
        tmp_path, monkeypatch, worker_failure):
  source = tmp_path / "source"
  (source / "src/mamba_forest").mkdir(parents=True)
  (source / "src/mamba_forest/model.py").write_text("# test source\n")
  root = tmp_path / "run"
  shared = fixture(str(source))
  calls = []

  def worker(command, **kwargs):
    def value(flag):
      return command[command.index(flag) + 1]
    assert kwargs["env"]["PYTHONPATH"] == str(source / "src")
    phase, output = value("--phase"), Path(value("--output"))
    if phase == "prepare":
      (root / "fixture").mkdir()
      (root / "fixture/fixture.json").write_text(json.dumps(shared))
      output.write_text(json.dumps(shared))
      return SimpleNamespace(returncode=0)
    mode, batch = value("--mode"), int(value("--repeat-batch"))
    calls.append((phase, mode, batch))
    entry = {"phase": phase, "mode": mode, "precision": value("--precision"),
             "batch_size": batch}
    report = measurement(entry, shared)
    report["phase"] = phase
    if worker_failure in ("oom", "other") and mode == "training" and batch == 2:
      error = "ResourceExhaustedError" if worker_failure == "oom" else "ValueError"
      output.write_text(json.dumps({"status": "failed", "error_type": error}))
      return SimpleNamespace(returncode=1)
    if phase == "profile":
      if worker_failure == "profile":
        return SimpleNamespace(returncode=1)
      report.pop("latencies_seconds")
      report.pop("tiles_per_second")
      report.pop("median_seconds")
      report.pop("memory")
      directory = Path(value("--profile-dir"))
      directory.mkdir(parents=True)
      (directory / "test.xplane.pb").write_bytes(b"trace")
      report["steps"] = 1
      report["optimizer_updates"] = 1 if mode == "training" else None
      report["profile"] = {"status": "captured", "steps": 1,
                           "directory": str(directory), "artifacts": [{
                               "path": "test.xplane.pb", "bytes": 5,
                               "sha256": hashlib.sha256(b"trace").hexdigest()}]}
    output.write_text(json.dumps(report))
    return SimpleNamespace(returncode=0)

  monkeypatch.setattr(STUDY.subprocess, "run", worker)
  arguments = ["--source-root", str(source), "--run-root", str(root),
               "--device", "/CPU:0", "--batch-sizes", "1", "2", "--rounds", "2",
               "--steps", "2", "--warmup", "1", "--profile-steps", "1",
               "--training-precision", "float32"]
  if worker_failure in ("other", "profile"):
    with pytest.raises(RuntimeError):
      STUDY.main(arguments)
    assert json.loads((root / "execution.json").read_text())["status"] == "failed"
    if worker_failure == "profile":
      assert json.loads((root / "study.json").read_text())["status"] == "timing_passed"
    return
  STUDY.main(arguments)
  result = json.loads((root / "study.json").read_text())
  assert result["status"] == "passed"
  assert result["gpu_trace_coverage"] == "not_applicable_on_cpu"
  assert [batch for phase, mode, batch in calls
          if phase == "measure" and mode == "inference"] == [1, 2, 2, 1]
  phases = [phase for phase, _, _ in calls]
  assert max(i for i, phase in enumerate(phases) if phase == "measure") < min(
      i for i, phase in enumerate(phases) if phase == "profile")
  if worker_failure == "oom":
    assert calls.count(("measure", "training", 2)) == 1
    assert ("profile", "training", 2) not in calls
    assert result["summary"]["training"]["batches"][1]["status"] == "out_of_memory"
  previous = copy.deepcopy(result)
  with pytest.raises(FileExistsError):
    STUDY.main(arguments)
  assert json.loads((root / "study.json").read_text()) == previous


def test_capture_artifacts_must_exist_and_keep_their_identity(tmp_path):
  directory = tmp_path / "profiles/profile-inference-b1"
  directory.mkdir(parents=True)
  path = directory / "trace.xplane.pb"
  path.write_bytes(b"trace")
  entry = {"file": "profile-inference-b1.json"}
  report = {"profile": {"directory": str(directory), "artifacts": [
      {"path": path.name, "bytes": 5,
       "sha256": hashlib.sha256(b"trace").hexdigest()}]}}
  STUDY.validate_profile_files(report, tmp_path, entry)
  path.write_bytes(b"other")
  with pytest.raises(ValueError, match="changed"):
    STUDY.validate_profile_files(report, tmp_path, entry)
  path.unlink()
  with pytest.raises(ValueError, match="Missing"):
    STUDY.validate_profile_files(report, tmp_path, entry)
