"""Batch replication and profiling keep traced work separate from timing."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/measure_revision.py"
SPEC = importlib.util.spec_from_file_location("revision_profile_worker", SCRIPT)
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


def test_batch_worker_and_training_profile_use_one_shared_fixture(tmp_path):
  fixture = tmp_path / "fixture"
  environment = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "2",
                 "TF_NUM_INTRAOP_THREADS": "2", "TF_NUM_INTEROP_THREADS": "2",
                 "OMP_NUM_THREADS": "2"}

  def run(*arguments, success=True):
    process = subprocess.run(
        [sys.executable, str(SCRIPT), "--fixture-dir", str(fixture), *arguments],
        env=environment, capture_output=True, text=True)
    assert (process.returncode == 0) is success, process.stdout + process.stderr
    return process

  run("--phase", "prepare", "--tile-size", "8", "--d-model", "4",
      "--d-state-temporal", "2", "--d-state-spatial", "2",
      "--bottleneck-channels", "8")
  metadata = json.loads((fixture / "fixture.json").read_text())
  timing_path = tmp_path / "timing.json"
  run("--phase", "measure", "--repeat-batch", "2", "--warmup", "1",
      "--steps", "2", "--output", str(timing_path))
  timing = json.loads(timing_path.read_text())
  assert timing["batch_size"] == 2
  assert timing["batch_replication"] == {
      "kind": "repeat_single_fixture_tile", "repeat_count": 2,
      "fixture_batch_size": 1}
  assert timing["input_shapes"]["s2"] == [2, 4, 8, 8, 10]
  assert timing["weights_sha256"] == metadata["weights_sha256"]
  assert timing["inputs_sha256"] == metadata["inputs_sha256"]
  assert timing["tiles_per_second"] == pytest.approx(
      4 / sum(timing["latencies_seconds"]))
  assert "profile" not in timing

  profile_path, trace_dir = tmp_path / "profile.json", tmp_path / "trace"
  run("--phase", "profile", "--mode", "training", "--repeat-batch", "2",
      "--warmup", "1", "--profile-steps", "2", "--profile-dir", str(trace_dir),
      "--output", str(profile_path))
  profile = json.loads(profile_path.read_text())
  assert profile["status"] == "passed" and profile["phase"] == "profile"
  assert profile["batch_size"] == 2 and profile["steps"] == 2
  assert profile["warmup_optimizer_updates"] == 1
  assert profile["optimizer_updates"] == 2
  assert len(profile["training_loss_samples"]) == 2
  assert profile["weights_sha256"] == metadata["weights_sha256"]
  assert profile["inputs_sha256"] == metadata["inputs_sha256"]
  assert profile["forward_agreement"]["passed"]
  assert not {"latencies_seconds", "tiles_per_second", "median_seconds",
              "memory"}.intersection(profile)
  capture = profile["profile"]
  assert capture["status"] == "captured"
  assert capture["gpu_trace_coverage"] == "not_applicable_on_cpu"
  assert {"temporal_mamba", "spatial_mamba", "optical_proj", "fusion_layer",
          "enc1", "dec1"} <= capture["layer_scopes"].keys()
  assert capture["tracer_options"] == {
      "host_tracer_level": 2, "python_tracer_level": 0, "device_tracer_level": 1}
  assert any(item["path"].endswith(".xplane.pb") and item["bytes"] > 0
             for item in capture["artifacts"])
  for item in capture["artifacts"]:
    trace = trace_dir / item["path"]
    assert trace.stat().st_size == item["bytes"]
    assert hashlib.sha256(trace.read_bytes()).hexdigest() == item["sha256"]
  original = profile_path.read_bytes()
  run("--phase", "profile", "--profile-dir", str(trace_dir),
      "--output", str(profile_path), success=False)
  assert profile_path.read_bytes() == original

  with np.load(fixture / "inputs.npz", allow_pickle=False) as saved:
    arrays = {key: np.repeat(saved[key], 2, axis=0) for key in saved.files}
  np.savez_compressed(fixture / "inputs.npz", **arrays)
  metadata["input_shapes"] = {key: list(value.shape)
                              for key, value in arrays.items()}
  metadata["inputs_sha256"] = WORKER.digest(fixture / "inputs.npz")
  (fixture / "fixture.json").write_text(json.dumps(metadata))
  rejected = tmp_path / "rejected.json"
  run("--phase", "measure", "--repeat-batch", "2", "--output", str(rejected),
      success=False)
  failure = json.loads(rejected.read_text())
  assert failure["error_type"] == "ValueError"
  assert "one-tile fixture" in failure["error"]


@pytest.mark.parametrize("arguments", [
    ["--phase", "profile"],
    ["--phase", "profile", "--output", "report.json"],
    ["--phase", "measure", "--output", "report.json", "--profile-dir", "trace"],
    ["--phase", "prepare", "--repeat-batch", "2"],
    ["--phase", "measure", "--output", "report.json", "--repeat-batch", "0"],
    ["--phase", "prepare", "--profile-steps", "0"],
    ["--phase", "prepare", "--device", "/GPU:-1"],
    ["--phase", "prepare", "--device", "/GPU:abc"],
])
def test_invalid_profile_and_batch_arguments_fail_before_tensorflow(arguments):
  with pytest.raises(SystemExit):
    WORKER.parse_args(["--fixture-dir", "fixture", *arguments])


@pytest.mark.parametrize("failure", [None, "step", "consume", "trace", "empty"])
def test_profiler_session_is_stopped_on_work_failure(tmp_path, failure):
  events = []
  trace_dir = tmp_path / "trace"

  def start(directory, options):
    events.append("start")
    assert directory == str(trace_dir)

  def stop():
    events.append("stop")
    if failure != "empty":
      (trace_dir / "test.xplane.pb").write_bytes(b"mock profile")

  def trace(name, **kwargs):
    events.append((name, kwargs))
    if failure == "trace":
      raise RuntimeError("trace failed")
    return contextlib.nullcontext()

  def step():
    events.append("step")
    if failure == "step":
      raise RuntimeError("step failed")
    return [1]

  def consume(values):
    assert values == [1]
    if failure == "consume":
      raise RuntimeError("update failed")

  fake = SimpleNamespace(profiler=SimpleNamespace(experimental=SimpleNamespace(
      start=start, stop=stop, Trace=trace,
      ProfilerOptions=lambda **options: options)))
  if failure is not None:
    with pytest.raises(RuntimeError):
      WORKER.capture_profile(fake, trace_dir, 2, step, consume, "training")
  else:
    result = WORKER.capture_profile(fake, trace_dir, 2, step, consume, "training")
    assert result["status"] == "captured" and result["steps"] == 2
    assert events.count("step") == 2
    assert ("training_step", {"step_num": 1, "_r": 1}) in events
  assert events[0] == "start" and events[-1] == "stop"


def test_existing_trace_directory_is_preserved(tmp_path):
  trace_dir = tmp_path / "trace"
  trace_dir.mkdir()
  marker = trace_dir / "previous.xplane.pb"
  marker.write_bytes(b"previous")
  with pytest.raises(FileExistsError):
    WORKER.capture_profile(None, trace_dir, 1, None, None, "inference")
  assert marker.read_bytes() == b"previous"


def test_failed_start_does_not_stop_an_unrelated_profiler_session(tmp_path):
  events = []

  def start(*args, **kwargs):
    raise RuntimeError("another profiler is already active")

  fake = SimpleNamespace(profiler=SimpleNamespace(experimental=SimpleNamespace(
      start=start, stop=lambda: events.append("stop"),
      ProfilerOptions=lambda **options: options)))
  with pytest.raises(RuntimeError, match="already active"):
    WORKER.capture_profile(fake, tmp_path / "trace", 1, None, None, "inference")
  assert events == []
