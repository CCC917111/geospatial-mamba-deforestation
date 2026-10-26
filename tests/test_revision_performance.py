"""Shared performance fixtures reject changed data and preserve prior reports."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "measure_revision.py"


def test_revision_harness_shared_fixture_updates_and_tamper_protection(tmp_path):
  fixture = tmp_path / "fixture"
  environment = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "2",
                 "TF_NUM_INTRAOP_THREADS": "2", "TF_NUM_INTEROP_THREADS": "2",
                 "OMP_NUM_THREADS": "2"}

  def run(*arguments, success=True):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--fixture-dir", str(fixture), *arguments],
        env=environment, text=True, capture_output=True)
    if success:
      assert result.returncode == 0, result.stdout + result.stderr
    else:
      assert result.returncode != 0
    return result

  run("--phase", "prepare", "--tile-size", "8", "--d-model", "4",
      "--d-state-temporal", "2", "--d-state-spatial", "2",
      "--bottleneck-channels", "8")
  metadata = json.loads((fixture / "fixture.json").read_text())
  for mode in ("inference", "training"):
    path = tmp_path / (mode + ".json")
    arguments = ("--phase", "measure", "--mode", mode, "--warmup", "1",
                 "--steps", "2", "--output", str(path))
    run(*arguments)
    report = json.loads(path.read_text())
    assert report["status"] == "passed"
    assert report["weights_sha256"] == metadata["weights_sha256"]
    assert report["inputs_sha256"] == metadata["inputs_sha256"]
    assert report["forward_agreement"]["passed"]
    assert len(report["latencies_seconds"]) == 2
    assert report["optimizer_updates"] == (2 if mode == "training" else None)
    assert report["memory"]["status"] == "not_applicable_on_cpu"
    original = path.read_bytes()
    run(*arguments, success=False)
    assert path.read_bytes() == original

  (fixture / "inputs.npz").write_bytes(b"changed shared inputs")
  failure = tmp_path / "changed-inputs.json"
  run("--phase", "measure", "--output", str(failure), success=False)
  report = json.loads(failure.read_text())
  assert report["status"] == "failed" and "Inputs changed" in report["error"]
