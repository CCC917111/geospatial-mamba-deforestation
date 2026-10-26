"""Performance summaries reject mismatched fixtures and incomplete workloads."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_performance.py"
SPEC = importlib.util.spec_from_file_location("revision_comparison", SCRIPT)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def _reports(root):
  fixture = {"weights_sha256": "same-weights", "inputs_sha256": "same-inputs",
             "model_config": {"num_classes": 9},
             "input_shapes": {"s2": [1, 4, 8, 8, 10], "elevation": [1, 8, 8, 3],
                              "climate": [1, 4, 14], "labels": [1, 8, 8]},
             "source": {"module_sha256": {"model.py": "baseline"}},
             "runtime": {"requested_device": "/CPU:0", "tensorflow_version": "test"}}
  (root / "fixture").mkdir()
  (root / "fixture/fixture.json").write_text(json.dumps(fixture))
  plan = {"rounds": 1, "steps": 2, "warmup": 1, "device": "/CPU:0",
          "batch_size": 1, "tile_size": 8,
          "source_roots": {"baseline": "/baseline", "candidate": "/candidate"},
          "measurements": []}
  for mode in ("inference", "training"):
    for revision in ("baseline", "candidate"):
      filename = mode + "-" + revision + ".json"
      entry = {"mode": mode, "precision": "float32", "revision": revision,
               "file": filename}
      plan["measurements"].append(entry)
      report = {**fixture, "status": "passed", "mode": mode,
                "precision": "float32", "steps": 2, "warmup_steps": 1,
                "forward_agreement": {"passed": True}, "optimizer_updates": 2,
                "source": {"module_sha256": {"model.py": revision},
                           "package_directory": f"/{revision}/src/mamba_forest"},
                "parameters": 123, "memory": {"status": "not_applicable_on_cpu"},
                "latencies_seconds": [0.03, 0.01], "median_seconds": 0.02}
      (root / filename).write_text(json.dumps(report))
  return plan


@pytest.mark.parametrize("field,value", [
    ("weights_sha256", "different"), ("inputs_sha256", "different"),
    ("model_config", {"num_classes": 2}), ("status", "failed"),
    ("optimizer_updates", 0), ("forward_agreement", {"passed": False}),
    ("runtime", {"requested_device": "/GPU:0", "tensorflow_version": "different"}),
    ("parameters", 456),
    ("latencies_seconds", [0.02]), ("median_seconds", 0.0001),
    ("source", {"package_directory": "/wrong/src/mamba_forest",
                "module_sha256": {"model.py": "candidate"}}),
])
def test_rejects_unmatched_or_incomplete_measurements(tmp_path, field, value):
  plan = _reports(tmp_path)
  assert comparison.compare(tmp_path, plan)["status"] == "passed"
  path = tmp_path / "training-candidate.json"
  report = json.loads(path.read_text())
  report[field] = value
  path.write_text(json.dumps(report))
  with pytest.raises(ValueError):
    comparison.compare(tmp_path, plan)


def test_rejects_inflated_batch_throughput(tmp_path):
  plan = _reports(tmp_path)
  plan["batch_size"] = 2
  with pytest.raises(ValueError, match="dimensions"):
    comparison.compare(tmp_path, plan)
