"""Benchmark summaries exclude warmup and first tracing without stale outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from mamba_forest import benchmark_predict


def _args(tmp_path, *flags):
  return benchmark_predict.parse_args([
      "--weights", str(tmp_path / "weights.h5"), "--input-dir", str(tmp_path / "tiles"),
      "--output-dir", str(tmp_path / "benchmark"), *flags])


def test_module_and_help_do_not_import_tensorflow(tmp_path):
  source = Path(benchmark_predict.__file__).resolve().parents[1]
  script = """import sys
from mamba_forest.benchmark_predict import parse_args
args=parse_args(['--weights','missing.h5','--input-dir','missing',
                 '--output-dir',sys.argv[1]])
assert args.repeats==30 and args.warmup==1
assert 'tensorflow' not in sys.modules
"""
  subprocess.run([sys.executable, "-c", script, str(tmp_path / "unused")], check=True,
                 env={**os.environ, "PYTHONPATH": str(source)})
  assert not (tmp_path / "unused").exists()


def test_percentile_uses_raw_samples_and_does_not_invent_empty_timings():
  summary = benchmark_predict.latency_summary([1.0, 2.0, 3.0])
  assert summary["seconds"] == [1.0, 2.0, 3.0]
  assert summary["median_seconds"] == 2
  assert summary["p95_seconds"] == pytest.approx(2.9)
  assert benchmark_predict.latency_summary([])["median_seconds"] is None
  with pytest.raises(ValueError):
    benchmark_predict.latency_summary([float("nan")])


def test_benchmark_uses_production_api_and_keeps_cold_sample_out_of_steady_stats(
      tmp_path, monkeypatch):
  from mamba_forest import predict
  args = _args(tmp_path, "--repeats", "2", "--warmup", "0")
  calls = []

  def production(arguments, **kwargs):
    calls.append(kwargs)
    fields = ("read_validate_seconds", "preprocess_seconds", "inference_sync_seconds",
              "postprocess_seconds", "array_image_export_seconds", "pipeline_seconds")
    samples = [{"phase": "measured", "first_inference": first,
                **{field: value for field in fields}}
               for first, value in ((True, 10.0), (False, 1.0))]
    report = {
        "samples": samples, "passes": [
            {"phase": "measured", "wall_seconds": 6.0, "csv_export_seconds": 0.1},
            {"phase": "measured", "wall_seconds": 4.0, "csv_export_seconds": 0.1}],
        "metadata": {"save_options": {"png": False, "probabilities": False}},
        "scope": {}, "total_wall_seconds": 11.0,
        "first_inference_sync_seconds": 10.0, "model_build_load_seconds": 0.5,
        "cold_start_to_first_prediction_seconds": 10.5,
    }
    arguments.timings.write_text(json.dumps(report))
    return report

  monkeypatch.setattr(predict, "run_prediction", production)
  report = benchmark_predict.run_benchmark(args)
  assert calls == [{"repeats": 2, "warmup": 0, "quiet": True}]
  assert report["measured_tile_count"] == 2 and report["steady_state_tile_count"] == 1
  assert report["steady_state_stages"]["inference_sync_seconds"]["seconds"] == [1.0]
  assert report["measured_tiles_per_second"] == 0.2
  assert json.loads((args.output_dir / "benchmark.json").read_text()) == report


def test_existing_output_files_are_rejected_before_prediction(tmp_path, monkeypatch):
  from mamba_forest import predict
  args = _args(tmp_path)
  args.output_dir.mkdir()
  retained = args.output_dir / "notes.txt"
  retained.write_text("preserve")
  monkeypatch.setattr(predict, "run_prediction", lambda *a, **k: pytest.fail("called"))
  with pytest.raises(ValueError, match="fresh or empty"):
    benchmark_predict.run_benchmark(args)
  assert retained.read_text() == "preserve"


def test_output_guard_normalizes_missing_prefix_and_parent_segments(tmp_path):
  args = _args(tmp_path)
  protected = tmp_path / "protected"
  protected.mkdir()
  retained = protected / "notes.txt"
  retained.write_text("preserve")
  missing = tmp_path / "missing"
  args.output_dir = missing / ".." / "protected"
  with pytest.raises(ValueError, match="fresh or empty"):
    benchmark_predict.run_benchmark(args)
  assert retained.read_text() == "preserve"
  assert list(protected.iterdir()) == [retained]
  assert not missing.exists()


@pytest.mark.parametrize("flags", [["--repeats", "0"], ["--warmup", "-1"]])
def test_invalid_repeat_budgets_fail_without_output_creation(tmp_path, flags):
  with pytest.raises(SystemExit):
    _args(tmp_path, *flags)
  assert not (tmp_path / "benchmark").exists()
