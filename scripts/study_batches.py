"""Measure batch scaling, then capture separate TensorFlow profiler traces.

One revision, shared initial weights and one synthetic tile repeated per batch.
Every measurement/capture runs in a fresh process. This is a capacity study,
not an architecture comparison, training-convergence test or real-data score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

HELPER = Path(__file__).with_name("measure_revision.py").resolve()


def save(path, value):
  path = Path(path)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
  temporary.replace(path)


def aggregate(reports, batch):
  values = [x for report in reports for x in report["latencies_seconds"]]
  ordered = sorted(values)
  position = 0.95 * (len(ordered) - 1)
  lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
  return {"median_seconds": statistics.median(values),
          "p95_seconds": ordered[lower] + (ordered[upper] - ordered[lower])
          * (position - lower), "samples": len(values),
          "tiles_per_second": batch * len(values) / sum(values),
          "round_medians_seconds": [x["median_seconds"] for x in reports],
          "round_tiles_per_second": [x["tiles_per_second"] for x in reports]}


def validate_report(report, fixture, plan, entry):
  """Reject inconsistent source, data, batch semantics and sample budgets."""
  expected_shapes = {key: [entry["batch_size"], *shape[1:]]
                     for key, shape in fixture["input_shapes"].items()}
  replication = {"kind": "repeat_single_fixture_tile",
                 "repeat_count": entry["batch_size"], "fixture_batch_size": 1}
  expected = {"status": "passed", "phase": entry["phase"],
              "mode": entry["mode"], "precision": entry["precision"],
              "batch_size": entry["batch_size"], "input_shapes": expected_shapes,
              "batch_replication": replication, "warmup_steps": plan["warmup"]}
  for key, value in expected.items():
    if report.get(key) != value:
      raise ValueError(f"Unmatched {key}: {entry['file']}")
  for key in ("source", "runtime", "model_config", "weights_sha256",
              "inputs_sha256"):
    if report.get(key) != fixture[key]:
      raise ValueError(f"Unmatched {key}: {entry['file']}")
  if not report.get("forward_agreement", {}).get("passed"):
    raise ValueError("Forward agreement did not pass")
  budget = plan["steps"] if entry["phase"] == "measure" else plan["profile_steps"]
  if entry["mode"] == "training" and report.get("optimizer_updates") != budget:
    raise ValueError("Incomplete optimizer update budget")
  if entry["phase"] == "profile":
    profile = report.get("profile", {})
    invalid = (profile.get("status") != "captured"
               or profile.get("steps") != budget or report.get("steps") != budget
               or not profile.get("artifacts")
               or any(key in report for key in (
                   "latencies_seconds", "median_seconds", "tiles_per_second",
                   "memory")))
    if invalid:
      raise ValueError("Profile artifacts are missing")
    return
  samples = report.get("latencies_seconds")
  if (report.get("steps") != budget or not isinstance(samples, list)
      or len(samples) != budget
      or any(isinstance(x, bool) or not isinstance(x, (int, float))
             or not math.isfinite(x) or x <= 0 for x in samples)):
    raise ValueError("Incomplete or invalid latency sample budget")
  if not math.isclose(report["median_seconds"], statistics.median(samples),
                      rel_tol=1e-12, abs_tol=1e-12):
    raise ValueError("Median differs from raw samples")
  expected_rate = entry["batch_size"] * budget / sum(samples)
  if not math.isclose(report["tiles_per_second"], expected_rate, rel_tol=1e-9):
    raise ValueError("Throughput differs from effective batch and raw samples")


def validate_profile_files(report, root, entry):
  profile = report["profile"]
  directory = root / "profiles" / Path(entry["file"]).stem
  invalid = (profile.get("directory") != str(directory)
             or directory.is_symlink() or not directory.is_dir())
  if invalid:
    raise ValueError("Profile directory differs from the selected capture")
  xplanes = 0
  for artifact in profile["artifacts"]:
    relative = Path(artifact["path"])
    if relative.is_absolute() or ".." in relative.parts:
      raise ValueError("Unsafe profiler artifact path")
    path = directory / relative
    invalid = (path.is_symlink() or not path.is_file()
               or not path.resolve().is_relative_to(directory.resolve()))
    if invalid:
      raise ValueError("Missing or unsafe profiler artifact")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
      for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    invalid = (path.stat().st_size != artifact["bytes"]
               or digest.hexdigest() != artifact["sha256"])
    if invalid:
      raise ValueError("Profiler artifact changed after capture")
    xplanes += path.name.endswith(".xplane.pb") and path.stat().st_size > 0
  if not xplanes:
    raise ValueError("No nonempty XPlane trace was captured")


def summarize(root, plan, fixture):
  results, parameters = {}, set()
  for mode in ("inference", "training"):
    cases = []
    for batch in plan["batch_sizes"]:
      entries = [x for x in plan["measurements"]
                 if x["mode"] == mode and x["batch_size"] == batch]
      reports, oom = [], False
      for entry in entries:
        path = root / entry["file"]
        if not path.is_file():
          continue
        report = json.loads(path.read_text())
        if report.get("error_type") == "ResourceExhaustedError":
          oom = True
          continue
        validate_report(report, fixture, plan, entry)
        parameters.add(report["parameters"])
        reports.append(report)
      if oom:
        cases.append({"batch_size": batch, "status": "out_of_memory",
                      "completed_rounds": len(reports)})
        continue
      if len(reports) != plan["rounds"]:
        raise ValueError(f"Incomplete {mode} batch {batch}")
      values = aggregate(reports, batch)
      values.update({"batch_size": batch, "status": "passed",
                     "completed_rounds": len(reports),
                     "amortized_median_seconds_per_tile": (
                         values["median_seconds"] / batch),
                     "allocator_memory_by_round": [x["memory"] for x in reports]})
      peaks = [x["memory"].get("peak_bytes") for x in reports]
      values["median_allocator_peak_bytes"] = (
          statistics.median(peaks) if all(x is not None for x in peaks) else None)
      cases.append(values)
    successful = [x for x in cases if x["status"] == "passed"]
    if not successful or successful[0]["batch_size"] != 1:
      raise ValueError(f"Batch-one reference is unavailable for {mode}")
    reference = successful[0]["tiles_per_second"]
    for case in successful:
      case["throughput_ratio_to_batch_one"] = case["tiles_per_second"] / reference
    results[mode] = {
        "batches": cases,
        "highest_observed_throughput_batch": max(
            successful, key=lambda x: x["tiles_per_second"])["batch_size"]}
  if len(parameters) != 1:
    raise ValueError("Model parameter counts differ between measurements")
  return {"schema_version": 1, "status": "timing_passed",
          "model_parameters": parameters.pop(), "summary": results,
          "fixture": fixture, "plan": plan,
          "scope": (
              "Synthetic batch-capacity study on one revision and one repeated "
              "tile. Batch latency and amortized time are distinct. Throughput "
              "excludes IO, preparation, loading and exports. Profiler captures "
              "are separate fresh processes and excluded from timing/memory "
              "samples. Memory is TensorFlow allocator usage, not total VRAM. "
              "No accuracy, convergence or statistical significance claim.")}


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source-root", type=Path, default=HELPER.parents[1])
  parser.add_argument("--run-root", type=Path, required=True)
  parser.add_argument("--device", default="/GPU:0")
  parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
  parser.add_argument("--rounds", type=int, default=3)
  parser.add_argument("--steps", type=int, default=30)
  parser.add_argument("--warmup", type=int, default=5)
  parser.add_argument("--training-precision", choices=("float32", "mixed_float16"),
                      default="mixed_float16")
  parser.add_argument("--profile-steps", type=int, default=3)
  parser.add_argument("--skip-profiles", action="store_true")
  parser.add_argument("--weights", type=Path)
  parser.add_argument("--tile-size", type=int, default=128)
  parser.add_argument("--d-model", type=int, default=32)
  parser.add_argument("--d-state-temporal", type=int, default=8)
  parser.add_argument("--d-state-spatial", type=int, default=16)
  parser.add_argument("--bottleneck-channels", type=int, default=512)
  args = parser.parse_args(argv)
  budgets = [args.rounds, args.steps, args.warmup, args.profile_steps,
             args.tile_size, args.d_model, args.d_state_temporal,
             args.d_state_spatial, args.bottleneck_channels, *args.batch_sizes]
  if min(budgets) < 1 or args.tile_size % 8:
    parser.error("budgets must be positive; tile size must be divisible by eight")
  if len(set(args.batch_sizes)) != len(args.batch_sizes) or 1 not in args.batch_sizes:
    parser.error("batch sizes must be distinct and include one")
  if not re.fullmatch(r"/(CPU|GPU):[0-9]+", args.device):
    parser.error("device must be an explicit /CPU:N or /GPU:N with nonnegative N")
  args.batch_sizes.sort()
  return args


def main(argv=None):
  args = parse_args(argv)
  source = args.source_root.expanduser().resolve()
  if not (source / "src/mamba_forest/model.py").is_file():
    raise ValueError("Source root does not contain the project model")
  root = args.run_root.expanduser().resolve()
  root.mkdir(parents=True, exist_ok=False)
  (root / "logs").mkdir()
  plan = {"schema_version": 1, "study_type": "batch_scaling",
          "source_root": str(source), "device": args.device,
          "batch_sizes": args.batch_sizes, "rounds": args.rounds,
          "steps": args.steps, "warmup": args.warmup,
          "profile_steps": args.profile_steps,
          "profiles_requested": not args.skip_profiles,
          "training_precision": args.training_precision,
          "tile_size": args.tile_size,
          "harness_sha256": hashlib.sha256(HELPER.read_bytes()).hexdigest(),
          "measurements": [], "profiles": []}
  for round_index in range(args.rounds):
    order = args.batch_sizes if round_index % 2 == 0 else args.batch_sizes[::-1]
    for mode, precision in (("inference", "float32"),
                            ("training", args.training_precision)):
      for batch in order:
        identifier = f"{mode}-b{batch}-{round_index:02d}"
        plan["measurements"].append({
            "phase": "measure", "mode": mode, "precision": precision,
            "batch_size": batch, "round": round_index,
            "file": identifier + ".json"})
  save(root / "plan.json", plan)
  execution = {"status": "running", "stages": []}

  def run(identifier, arguments):
    command = [sys.executable, str(HELPER), "--fixture-dir", str(root / "fixture"),
               "--device", args.device, *arguments]
    stage = {"id": identifier, "command": command, "status": "running"}
    execution["stages"].append(stage)
    save(root / "execution.json", execution)
    print("Running", identifier, flush=True)
    started = time.perf_counter()
    with (root / "logs" / (identifier + ".log")).open("x") as log:
      completed = subprocess.run(
          command, cwd=source,
          env={**os.environ, "PYTHONPATH": str(source / "src")},
          stdout=log, stderr=subprocess.STDOUT, check=False)
    stage.update({"returncode": completed.returncode,
                  "process_seconds": time.perf_counter() - started,
                  "status": "passed" if completed.returncode == 0 else "failed"})
    save(root / "execution.json", execution)
    return completed.returncode

  try:
    preparation = ["--phase", "prepare", "--batch-size", "1", "--tile-size",
                   str(args.tile_size), "--precisions", "float32"]
    if args.training_precision != "float32":
      preparation.append(args.training_precision)
    for name in ("d_model", "d_state_temporal", "d_state_spatial",
                 "bottleneck_channels"):
      preparation += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    preparation += ["--output", str(root / "preparation.json")]
    if args.weights:
      preparation += ["--weights", str(args.weights.expanduser().resolve())]
    if run("prepare", preparation):
      raise RuntimeError("Fixture preparation failed; inspect logs/prepare.log")
    fixture = json.loads((root / "fixture/fixture.json").read_text())
    expected_package = str(source / "src/mamba_forest")
    invalid = (fixture["source"]["package_directory"] != expected_package
               or fixture["runtime"]["requested_device"] != args.device)
    if invalid:
      raise ValueError("Fixture used a different source or device")
    blocked = set()
    for entry in plan["measurements"]:
      key = (entry["mode"], entry["batch_size"])
      identifier = Path(entry["file"]).stem
      if key in blocked:
        execution["stages"].append({"id": identifier,
                                    "status": "skipped_after_out_of_memory"})
        save(root / "execution.json", execution)
        continue
      arguments = ["--phase", "measure", "--mode", entry["mode"],
                   "--precision", entry["precision"], "--repeat-batch",
                   str(entry["batch_size"]), "--steps", str(args.steps),
                   "--warmup", str(args.warmup), "--output",
                   str(root / entry["file"])]
      code = run(identifier, arguments)
      if code:
        report_path = root / entry["file"]
        failure = json.loads(report_path.read_text()) if report_path.is_file() else {}
        exhausted = (failure.get("status") == "failed"
                     and failure.get("error_type") == "ResourceExhaustedError"
                     and entry["batch_size"] != 1)
        if exhausted:
          blocked.add(key)
          execution["stages"][-1]["status"] = "out_of_memory"
          save(root / "execution.json", execution)
          continue
        raise RuntimeError(f"{identifier} failed; inspect its report and log")
    report = summarize(root, plan, fixture)
    save(root / "study.json", report)
    if not args.skip_profiles:
      for mode in ("inference", "training"):
        complete = [x["batch_size"] for x in report["summary"][mode]["batches"]
                    if x["status"] == "passed"]
        for batch in sorted({1, max(complete)}):
          identifier = f"profile-{mode}-b{batch}"
          precision = "float32" if mode == "inference" else args.training_precision
          entry = {"phase": "profile", "mode": mode, "precision": precision,
                   "batch_size": batch, "file": identifier + ".json"}
          plan["profiles"].append(entry)
          save(root / "plan.json", plan)
          if run(identifier, ["--phase", "profile", "--mode", mode,
                              "--precision", precision, "--repeat-batch", str(batch),
                              "--warmup", str(args.warmup), "--profile-steps",
                              str(args.profile_steps), "--profile-dir",
                              str(root / "profiles" / identifier), "--output",
                              str(root / entry["file"])]):
            raise RuntimeError(
                f"{identifier} failed; timed results remain in study.json")
          capture = json.loads((root / entry["file"]).read_text())
          validate_report(capture, fixture, plan, entry)
          validate_profile_files(capture, root, entry)
    if hashlib.sha256(HELPER.read_bytes()).hexdigest() != plan["harness_sha256"]:
      raise ValueError("Measurement harness changed during the study")
    coverage = ("not_collected" if args.skip_profiles else
                "not_applicable_on_cpu" if args.device.startswith("/CPU:") else
                "unverified_in_viewer")
    report.update({"status": "passed", "plan": plan,
                   "profiles": plan["profiles"], "gpu_trace_coverage": coverage})
    save(root / "study.json", report)
    execution["status"] = "passed"
    print(json.dumps(report["summary"], indent=2))
  except BaseException as error:
    execution.update({"status": "interrupted" if isinstance(error, KeyboardInterrupt)
                      else "failed", "error": f"{type(error).__name__}: {error}"})
    raise
  finally:
    save(root / "execution.json", execution)


if __name__ == "__main__":
  main()
