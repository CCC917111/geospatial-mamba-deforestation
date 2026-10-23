"""Benchmark the production NPZ-to-prediction path with bounded tile memory.

Each warmup or measured repeat is a complete input pass, including exports.
The model and compiled predictor are reused. The default records 30 measured
passes after one warmup pass. Source caches are not cleared. This measures
performance only; it establishes no accuracy or GPU performance claims.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

from mamba_forest import checkpoint


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--weights", required=True)
  parser.add_argument("--input-dir", required=True)
  parser.add_argument("--output-dir", required=True, type=Path,
                      help="Fresh or empty directory for outputs and JSON reports")
  parser.add_argument("--pattern", default="*.npz")
  parser.add_argument("--repeats", type=int, default=30,
                      help="Measured input passes with distinct output directories")
  parser.add_argument("--warmup", type=int, default=1,
                      help="Warmup passes, exported and recorded but not measured")
  parser.add_argument("--save-png", action="store_true")
  parser.add_argument("--save-probabilities", action="store_true")
  checkpoint.add_model_arguments(parser)
  args = parser.parse_args(argv)
  if args.repeats < 1 or args.warmup < 0:
    parser.error("--repeats must be positive and --warmup must be non-negative")
  try:
    checkpoint.resolve_model_arguments(args)
  except (ValueError, OSError) as error:
    parser.error(str(error))
  return args


def latency_summary(values):
  """Returns raw seconds, median and linearly interpolated p95."""
  values = list(values)
  if not values:
    return {"count": 0, "seconds": [], "median_seconds": None, "p95_seconds": None}
  if any(not math.isfinite(value) or value < 0 for value in values):
    raise ValueError("timing samples must be finite and non-negative")
  ordered = sorted(values)
  position = 0.95 * (len(ordered) - 1)
  lower = int(position)
  upper = min(lower + 1, len(ordered) - 1)
  p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
  return {"count": len(values), "seconds": values,
          "median_seconds": statistics.median(values), "p95_seconds": p95}


def run_benchmark(args):
  """Runs one initialized predictor through warmup and measured input passes."""
  started = time.perf_counter()
  root = Path(args.output_dir).expanduser().resolve()
  if root.exists() and (not root.is_dir() or any(root.iterdir())):
    raise ValueError("--output-dir must be a fresh or empty directory")
  args.output_dir = root
  root.mkdir(parents=True, exist_ok=True)
  with (root / ".benchmark-started").open("x", encoding="utf-8") as handle:
    handle.write("Reserved for one benchmark execution.\n")
  framework_already_loaded = "tensorflow" in sys.modules
  import_started = time.perf_counter()
  from mamba_forest import predict
  framework_import_seconds = time.perf_counter() - import_started
  args.timings = root / "prediction_timings.json"
  prediction = predict.run_prediction(
      args, repeats=args.repeats, warmup=args.warmup, quiet=True)
  measured = [sample for sample in prediction["samples"]
              if sample["phase"] == "measured"]
  steady = [sample for sample in measured if not sample["first_inference"]]
  measured_passes = [item for item in prediction["passes"]
                     if item["phase"] == "measured"]
  wall = sum(item["wall_seconds"] for item in measured_passes)
  fields = ("read_validate_seconds", "preprocess_seconds", "inference_sync_seconds",
            "postprocess_seconds", "array_image_export_seconds", "pipeline_seconds")
  report = {
      "schema_version": 1, "metadata": prediction["metadata"],
      "scope": {
          **prediction["scope"],
          "start": ("run_benchmark entry; excludes process startup, parsing "
                    "and config resolution"),
          "framework_import": ("timed separately; cached when already "
                               "imported by caller"),
          "steady_state": ("measured tiles excluding the first inference "
                           "even when warmup is zero"),
          "console": ("per-tile console output disabled; "
                      "all prediction file exports retained"),
          "report_write": ("prediction report write included in benchmark total; "
                           "final benchmark JSON write excluded"),
      },
      "repeats": args.repeats, "warmup_passes": args.warmup,
      "framework_already_loaded": framework_already_loaded,
      "framework_import_seconds": framework_import_seconds,
      "prediction_total_wall_seconds": prediction["total_wall_seconds"],
      "first_inference_sync_seconds": prediction["first_inference_sync_seconds"],
      "model_build_load_seconds": prediction["model_build_load_seconds"],
      "cold_start_to_first_prediction_seconds":
          prediction["cold_start_to_first_prediction_seconds"],
      "measured_tile_count": len(measured), "steady_state_tile_count": len(steady),
      "measured_pass_wall_seconds": wall,
      "measured_tiles_per_second": len(measured) / wall if wall > 0 else None,
      "steady_state_stages": {
          field: latency_summary(sample[field] for sample in steady)
          for field in fields},
      "csv_export": latency_summary(
          item["csv_export_seconds"] for item in measured_passes),
      "samples": prediction["samples"], "passes": prediction["passes"],
      "prediction_report": str(args.timings),
  }
  report["benchmark_total_wall_seconds"] = time.perf_counter() - started
  with (root / "benchmark.json").open("x", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2, allow_nan=False)
    handle.write("\n")
  return report


def main(argv=None):
  args = parse_args(argv)
  report = run_benchmark(args)
  print(f"Benchmark: {Path(args.output_dir) / 'benchmark.json'}")
  print(f"Measured tiles: {report['measured_tile_count']}; "
        f"steady-state tiles: {report['steady_state_tile_count']}")


if __name__ == "__main__":
  main()
