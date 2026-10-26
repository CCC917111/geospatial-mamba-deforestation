"""Compare two local source checkouts with one shared performance harness.

Each workload/revision/round runs in a fresh Python process. The baseline
creates the synthetic inputs, checkpoint and reference predictions once.
No source is changed, no dataset is downloaded and no accuracy is measured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


HELPER = Path(__file__).with_name("measure_revision.py").resolve()


def save(path, value):
  path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def aggregate(reports, batch_size):
  values = [value for report in reports for value in report["latencies_seconds"]]
  if not values or any(not math.isfinite(value) or value <= 0 for value in values):
    raise ValueError("Invalid latency samples")
  ordered = sorted(values)
  position = 0.95 * (len(ordered) - 1)
  lower = int(position)
  upper = min(lower + 1, len(ordered) - 1)
  return {"median_seconds": statistics.median(values),
          "p95_seconds": ordered[lower] + (ordered[upper] - ordered[lower])
          * (position - lower),
          "tiles_per_second": batch_size * len(values) / sum(values),
          "samples": len(values),
          "round_medians_seconds": [report["median_seconds"] for report in reports]}


def compare(root, plan):
  fixture = json.loads((root / "fixture" / "fixture.json").read_text())
  groups = {}
  fingerprints = {}
  for entry in plan["measurements"]:
    report = json.loads((root / entry["file"]).read_text())
    if report.get("status") != "passed":
      raise ValueError(f"Incomplete measurement: {entry['file']}")
    for key in ("weights_sha256", "inputs_sha256", "model_config"):
      if report[key] != fixture[key]:
        raise ValueError(f"Unmatched {key}: {entry['file']}")
    invalid = (report["mode"] != entry["mode"]
               or report["precision"] != entry["precision"]
               or report["steps"] != plan["steps"]
               or report["warmup_steps"] != plan["warmup"]
               or report["runtime"] != fixture["runtime"]
               or report["runtime"]["requested_device"] != plan["device"]
               or not report["forward_agreement"]["passed"])
    if invalid:
      raise ValueError(f"Unmatched measurement protocol: {entry['file']}")
    if entry["mode"] == "training" and report["optimizer_updates"] != plan["steps"]:
      raise ValueError("Training measurement skipped optimizer updates")
    source = report["source"]["module_sha256"]
    previous = fingerprints.setdefault(entry["revision"], source)
    if source != previous:
      raise ValueError("Source modules changed between rounds")
    groups.setdefault((entry["mode"], entry["revision"]), []).append(report)
  summaries = {}
  for mode in ("inference", "training"):
    baseline = groups[(mode, "baseline")]
    candidate = groups[(mode, "candidate")]
    if len(baseline) != plan["rounds"] or len(candidate) != plan["rounds"]:
      raise ValueError("Missing measurement rounds")
    if baseline[0]["parameters"] != candidate[0]["parameters"]:
      raise ValueError("Model parameter counts differ")
    left = aggregate(baseline, plan["batch_size"])
    right = aggregate(candidate, plan["batch_size"])
    ratios = [a["median_seconds"] / b["median_seconds"]
              for a, b in zip(baseline, candidate)]
    summaries[mode] = {
        "baseline": left, "candidate": right,
        "median_speedup": left["median_seconds"] / right["median_seconds"],
        "latency_reduction_percent": (
            100 * (1 - right["median_seconds"] / left["median_seconds"])),
        "paired_round_speedups": ratios,
        "paired_round_speedup_sample_sd": (
            statistics.stdev(ratios) if len(ratios) > 1 else None),
        "allocator_memory_by_round": {
            "baseline": [report["memory"] for report in baseline],
            "candidate": [report["memory"] for report in candidate]},
    }
  return {"schema_version": 1, "status": "passed", "plan": plan,
          "fixture": fixture, "module_sha256": fingerprints, "summary": summaries,
          "scope": (
              "Descriptive synchronized synthetic-compute comparison. "
              "Alternating order reduces ordering bias; rounds do not establish "
              "statistical significance, accuracy or another device's speed.")}


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--baseline-root", type=Path, required=True)
  parser.add_argument("--candidate-root", type=Path, required=True)
  parser.add_argument("--run-root", type=Path, required=True)
  parser.add_argument("--device", default="/CPU:0")
  parser.add_argument("--training-precision", choices=("float32", "mixed_float16"),
                      default="float32")
  parser.add_argument("--rounds", type=int, default=3)
  parser.add_argument("--steps", type=int, default=30)
  parser.add_argument("--warmup", type=int, default=5)
  parser.add_argument("--tile-size", type=int, default=128)
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--weights", type=Path)
  args = parser.parse_args(argv)
  if min(args.rounds, args.steps, args.warmup, args.tile_size, args.batch_size) < 1:
    parser.error("All budgets must be positive")
  sources = {"baseline": args.baseline_root.resolve(),
             "candidate": args.candidate_root.resolve()}
  for source in sources.values():
    if not (source / "src/mamba_forest/model.py").is_file():
      parser.error(f"Missing model source under {source}")
  root = args.run_root.expanduser().resolve()
  root.mkdir(parents=True, exist_ok=False)
  (root / "logs").mkdir()
  plan = {"schema_version": 1, "source_roots": {k: str(v) for k, v in sources.items()},
          "device": args.device, "training_precision": args.training_precision,
          "rounds": args.rounds, "steps": args.steps, "warmup": args.warmup,
          "tile_size": args.tile_size, "batch_size": args.batch_size,
          "harness_sha256": hashlib.sha256(HELPER.read_bytes()).hexdigest(),
          "measurements": []}
  save(root / "plan.json", plan)
  status = {"status": "running", "stages": []}

  def run(revision, identifier, arguments):
    command = [sys.executable, str(HELPER), "--fixture-dir", str(root / "fixture"),
               "--device", args.device, *arguments]
    entry = {"id": identifier, "revision": revision, "command": command}
    status["stages"].append(entry)
    save(root / "execution.json", status)
    print("Running", identifier, flush=True)
    started = time.perf_counter()
    with (root / "logs" / (identifier + ".log")).open("x") as log:
      result = subprocess.run(command, cwd=sources[revision],
                              env={**os.environ, "PYTHONPATH": str(
                                  sources[revision] / "src")},
                              stdout=log, stderr=subprocess.STDOUT, check=False)
    entry.update({"process_seconds": time.perf_counter() - started,
                  "returncode": result.returncode})
    if result.returncode:
      raise RuntimeError(f"{identifier} failed; see its log in {root / 'logs'}")

  try:
    precisions = list(dict.fromkeys(("float32", args.training_precision)))
    preparation = ["--phase", "prepare", "--tile-size", str(args.tile_size),
                   "--batch-size", str(args.batch_size), "--precisions", *precisions,
                   "--output", str(root / "preparation.json")]
    if args.weights:
      preparation += ["--weights", str(args.weights.resolve())]
    run("baseline", "prepare", preparation)
    for round_index in range(args.rounds):
      order = ("baseline", "candidate") if round_index % 2 == 0 else (
          "candidate", "baseline")
      for mode, precision in (("inference", "float32"),
                              ("training", args.training_precision)):
        for revision in order:
          identifier = f"{mode}-{round_index:02d}-{revision}"
          relative = identifier + ".json"
          plan["measurements"].append({"revision": revision, "mode": mode,
                                       "precision": precision, "round": round_index,
                                       "file": relative})
          save(root / "plan.json", plan)
          run(revision, identifier, [
              "--phase", "measure", "--mode", mode, "--precision", precision,
              "--steps", str(args.steps), "--warmup", str(args.warmup),
              "--output", str(root / relative)])
    summary = compare(root, plan)
    save(root / "comparison.json", summary)
    status["status"] = "passed"
    print(json.dumps(summary["summary"], indent=2))
  except BaseException as error:
    status.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
    raise
  finally:
    save(root / "execution.json", status)


if __name__ == "__main__":
  main()
