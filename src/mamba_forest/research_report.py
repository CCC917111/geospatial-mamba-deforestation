"""Validate a retained architecture pilot and report numeric class-ID scores.

The public ForTy integer-to-name mapping and Unknown evaluation protocol are
not yet confirmed. This module therefore reports only metrics over IDs 0..8.
It imports no ML framework and preserves all existing comparator checks.
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
import tempfile

from mamba_forest import compare
from mamba_forest.constants import GCS_ROOT, NUM_CLASSES, TOTAL_SHARDS


ARMS = ("cnn_ssm", "cnn_only")
METRICS = ("macro_f1_all_ids", "mean_iou_present_ids", "pixel_accuracy")
_TRAIN_FILES = (
    "best.weights.h5", "last.weights.h5", "config.json", "training_results.csv",
    "split_manifest.json", "checkpoint_selection.json", "test/test_metrics.json")
_OPTIONAL_RUN_FILES = ("train_wall_seconds.txt", "test_wall_seconds.txt")
_MUTABLE_NAMES = {
    "execution.json", "state.json", "research_state.json", "session.json",
    "checkpoint_inventory.json", "research_report.json", "checkpoint_manifest.json",
    "comparison.json"}
_RESERVED_OUTPUT_NAMES = _MUTABLE_NAMES - {"research_report.json"} | {
    "research_spec.json", "plan.json", "source_identity.json", "runtime_identity.json",
    "data_identity.json", "selected_data.json", "config.json",
    "split_manifest.json", "checkpoint_selection.json", "test_metrics.json"}


def _check(condition, message):
  if not condition:
    raise ValueError(message)


def _integer(value, label, minimum=0):
  _check(isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
         f"{label} must be an integer >= {minimum}")
  return value


def _json(path):
  def reject(value):
    raise ValueError(f"Non-finite JSON constant: {value}")
  value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)
  _check(isinstance(value, dict), f"Expected a JSON object: {path}")
  return value


def _digest(path):
  value = hashlib.sha256()
  with Path(path).open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      value.update(block)
  return value.hexdigest()


def _atomic_json(path, value):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = None
  try:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8",
                                     delete=False, prefix=".report-") as handle:
      temporary = Path(handle.name)
      json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
      handle.write("\n")
    os.replace(temporary, path)
  finally:
    if temporary is not None:
      temporary.unlink(missing_ok=True)


def _stats(values):
  return {"mean": statistics.mean(values),
          "sample_sd": statistics.stdev(values) if len(values) > 1 else None}


def _scores(matrix):
  """Recompute scores from numeric IDs without publishing semantic labels."""
  _check(isinstance(matrix, list) and len(matrix) == NUM_CLASSES,
         "Expected one confusion row per numeric class ID")
  for row in matrix:
    _check(isinstance(row, list) and len(row) == NUM_CLASSES,
           "Expected a square numeric-ID confusion matrix")
    for value in row:
      _integer(value, "confusion count")
  support = [sum(row) for row in matrix]
  predicted = [sum(row[index] for row in matrix) for index in range(NUM_CLASSES)]
  total = sum(support)
  _check(total > 0, "Confusion matrix is empty")
  classes = []
  for index in range(NUM_CLASSES):
    denominator = support[index] + predicted[index]
    union = denominator - matrix[index][index]
    f1 = 2 * matrix[index][index] / denominator if denominator else 0.0
    iou = matrix[index][index] / union if union else None
    classes.append({"id": index, "support": support[index],
                    "predicted_pixels": predicted[index], "f1_percent": 100 * f1,
                    "iou_percent": 100 * iou if iou is not None else None})
  present_iou = [item["iou_percent"] for item in classes
                 if item["iou_percent"] is not None]
  metrics = {
      "macro_f1_all_ids": statistics.mean(item["f1_percent"] for item in classes),
      "mean_iou_present_ids": statistics.mean(present_iou),
      "pixel_accuracy": 100 * sum(matrix[i][i] for i in range(NUM_CLASSES)) / total,
  }
  return metrics, classes


def _selection_seed(config):
  value = config.get("selection_seed")
  _check(value is not None,
         "Research reports require an explicit shared selection_seed")
  _integer(value, "selection_seed")
  _check(value <= 2**32 - 1, "selection_seed exceeds uint32 range")
  return value


def _updates(rows):
  keys = ("train_batches", "optimizer_updates", "skipped_optimizer_updates")
  present = [any(key in row for key in keys) for row in rows]
  if not any(present):
    return {"status": "not_recorded_in_legacy_history"}
  _check(all(present) and all(all(key in row for key in keys) for row in rows),
         "Optimizer-update history is incomplete")
  totals = {}
  for key in keys:
    values = [row[key] for row in rows]
    _check(all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(value) and int(value) == value and value >= 0
               for value in values), "Invalid optimizer-update counts")
    totals[key] = int(sum(values))
  _check(totals["optimizer_updates"] > 0
         and totals["optimizer_updates"] + totals["skipped_optimizer_updates"]
         == totals["train_batches"], "Optimizer update totals are inconsistent")
  return {"status": "recorded", **totals}


def _receipt(path, root, kind):
  path = Path(path)
  _check(path.is_file() and not path.is_symlink() and path.stat().st_size > 0,
         f"Missing, empty or symlinked retained file: {path}")
  resolved = path.resolve()
  if path.is_relative_to(root):
    _check(resolved.is_relative_to(root), f"Retained path escapes run root: {path}")
  try:
    name = path.relative_to(root).as_posix()
    external = False
  except ValueError:
    name, external = str(resolved), True
  return {"path": name, "external": external, "kind": kind,
          "bytes": path.stat().st_size, "sha256": _digest(path)}


def _validate_data_manifest(path, runs, shared):
  _check(path is not None, "A staged data manifest is required")
  path = Path(path).expanduser().resolve()
  manifest = _json(path)
  _check(manifest.get("schema_version") == 1 and manifest.get("status") == "ready",
         "Staged data manifest is incomplete or unsupported")
  config = manifest.get("config", {})
  first = runs[0]["config"]
  expected = {"total_shards": first["total_shards"],
              "selection_seed": shared["selection_seed"],
              "train_shards": first["train_shards"], "val_shards": first["val_shards"],
              "test_shards": runs[0]["metrics"]["test_shards"]}
  _check(all(config.get(key) == value for key, value in expected.items()),
         "Staged data configuration differs from the training/test protocol")
  source = config.get("source_root")
  cache = config.get("cache_root")
  _check(isinstance(source, str) and source and isinstance(cache, str) and cache,
         "Data source/cache roots are missing")
  roots_match = ("://" not in cache
                 and Path(cache).resolve() == Path(first["gcs_root"]).resolve())
  _check(roots_match,
         "Training data root differs from the staged local cache")
  expected_paths = [(split, Path(name).name) for split in (
      "train", "validation", "test") for name in shared["paths"][split]]
  files = manifest.get("files")
  _check(isinstance(files, list)
         and [(entry.get("split"), entry.get("filename")) for entry in files]
         == expected_paths, "Staged data file membership differs from the fixed cohort")
  for entry in files:
    filename = entry["filename"]
    valid_name = (isinstance(filename, str) and re.fullmatch(
        r"forty_v1-(train|validation|test)\.tfrecord-\d{5}-of-\d{5}", filename))
    _check(valid_name, "Unsafe or unsupported staged filename")
    _check(entry.get("source") == source.rstrip("/") + "/" + filename,
           "Staged source path differs from the source root")
    size = _integer(entry.get("bytes"), "staged file bytes", 1)
    digest = entry.get("sha256")
    _check(isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest),
           "Staged data digest is invalid")
    local = Path(cache) / filename
    _check(local.is_file() and not local.is_symlink()
           and local.resolve().parent == Path(cache).resolve(),
           f"Missing or unsafe staged data file: {filename}")
    _check(local.stat().st_size == size and _digest(local) == digest,
           f"Staged data bytes changed: {filename}")
  return manifest, path


def _immutable_receipts(root):
  """Select source/runtime receipts without hashing a live runner ledger."""
  paths = set()
  for path in root.iterdir():
    selected = (path.is_file() and path.name not in _MUTABLE_NAMES
                and path.suffix in (".json", ".txt", ".csv"))
    if selected:
      paths.add(path)
  for directory in ("sessions", "provenance", "receipts", "logs"):
    parent = root / directory
    if not parent.is_dir():
      continue
    for path in parent.rglob("*"):
      if not path.is_file() or path.suffix not in (".json", ".txt", ".csv", ".log"):
        continue
      name = path.name.lower()
      mutable = (name in _MUTABLE_NAMES or "state" in name
                 or name.startswith("session") or "report" in name or "compare" in name)
      if mutable:
        continue
      completed_log = name.startswith((
          "train", "evaluate", "preflight", "runtime_probe"))
      if directory == "logs" and not completed_log:
        continue
      paths.add(path)
  return sorted(paths)


def _real_provenance(root, data, data_path, runs):
  """Check immutable runner receipts before naming a real GPU pilot complete."""
  source = _json(root / "source_identity.json")
  runtime = _json(root / "runtime_identity.json")
  identity = _json(root / "data_identity.json")
  commit = source.get("commit")
  modules = source.get("module_sha256")
  _check(isinstance(commit, str) and re.fullmatch("[0-9a-f]{40}", commit)
         and isinstance(modules, dict) and modules
         and all(isinstance(name, str) and name and isinstance(value, str)
                 and re.fullmatch("[0-9a-f]{64}", value)
                 for name, value in modules.items()), "Invalid source identity receipt")
  versions = ("tensorflow", "keras", "numpy", "python")
  _check(all(isinstance(runtime.get(key), str) and runtime[key] for key in versions),
         "Invalid runtime identity receipt")
  probe = runtime.get("probe_device")
  _check(isinstance(probe, str) and probe.endswith("GPU:0"),
         "Real GPU pilot requires a recorded GPU:0 runtime probe")
  _check(all(run["config"]["tensorflow_version"] == runtime["tensorflow"]
             for run in runs), "Training version differs from runtime identity")
  digest = _digest(data_path)
  _check(identity.get("manifest_sha256") == digest
         and identity.get("config") == data["config"]
         and identity.get("files") == data["files"]
         and identity.get("data_kind") == "real_forty",
         "Data identity receipt differs from staged data manifest")
  return {"status": "passed", "source_commit": commit,
          "runtime_versions": {key: runtime[key] for key in versions},
          "gpu_probe_device": probe, "data_manifest_sha256": digest}


def build_report(run_root, seeds=(42, 43, 44), *, data_manifest=None, output=None):
  """Validate completed artifacts and atomically write a neutral pilot report.

  A valid staged data manifest is mandatory. Real-data completion is asserted
  only when the runner's immutable spec explicitly identifies official ForTy;
  synthetic checks and unverified origins receive different statuses.
  """
  root = Path(run_root).expanduser().resolve()
  destination = (Path(output).expanduser().resolve() if output else
                 root / "research_report.json")
  valid_output = (destination.suffix == ".json"
                  and destination.name not in _RESERVED_OUTPUT_NAMES)
  _check(valid_output,
         "Report output must be a .json file distinct from retained inputs and ledgers")
  if destination.exists():
    _check(destination == root / "research_report.json"
           and _json(destination).get("report_kind") == "neutral_architecture_pilot",
           "Custom output must be fresh; existing non-report files are preserved")
  seeds = list(seeds)
  spec_path = root / "research_spec.json"
  spec = _json(spec_path) if spec_path.is_file() else {}
  specified = spec.get("config", {}).get("seeds")
  if specified is not None:
    _check(seeds == specified, "Reported seeds differ from the frozen research spec")
  plan_path = root / "plan.json"
  if plan_path.is_file():
    planned = _json(plan_path).get("config", {}).get("seeds")
    if planned is not None:
      _check(seeds == planned, "Reported seeds differ from the frozen research plan")
  validated = compare.summarize_runs(root, seeds, comparison="architecture")
  pairs, runs, shared, budgets = [], [], None, None
  retained, checkpoint_records = {}, []
  for seed in seeds:
    pair = {"seed": seed}
    for arm in ARMS:
      directory = root / f"{arm}_{seed}"
      config = _json(directory / "config.json")
      manifest = _json(directory / "split_manifest.json")
      metrics = _json(directory / "test/test_metrics.json")
      rows = compare._read_history(directory / "training_results.csv", config["epochs"])
      selection_seed = _selection_seed(config)
      cohort = {"selection_seed": selection_seed, "paths": {
          "train": manifest["epochs"][0]["paths"],
          "validation": manifest["validation_paths"], "test": metrics["shard_paths"]}}
      examples = [row["train_examples"] for row in rows]
      if shared is None:
        shared, budgets = cohort, examples
      _check(cohort == shared, "Research seeds/arms do not share one fixed cohort")
      _check(examples == budgets, "Research seeds/arms have different example budgets")
      scores, classes = _scores(metrics["confusion_matrix"])
      parameter_count = metrics.get("parameters")
      _integer(parameter_count, "parameter count", 1)
      _check(parameter_count == validated["summary"][arm]["parameter_count"],
             "Parameter count differs from the validated comparison")
      selected = _json(directory / "checkpoint_selection.json")
      pair[arm] = {
          "metrics_percent": scores, "per_numeric_class": classes,
          "confusion_matrix": metrics["confusion_matrix"],
          "test_pixels": metrics["num_pixels"], "parameter_count": parameter_count,
          "train_examples": sum(examples), "optimizer_updates": _updates(rows),
          "validation_selection": {
              "criterion": "macro_f1_all_nine_ids", "mode": "max", "tie_break": "first",
              "epoch": selected["epoch"], "score_fraction": selected["score"]},
      }
      runs.append({"config": config, "metrics": metrics})
      for name in _TRAIN_FILES:
        kind = ("selected_checkpoint" if name == "best.weights.h5" else
                "last_weights_only" if name == "last.weights.h5" else "run_artifact")
        receipt = _receipt(directory / name, root, kind)
        retained[receipt["path"]] = receipt
        if name in ("best.weights.h5", "last.weights.h5"):
          checkpoint_records.append({"seed": seed, "model_variant": arm, **receipt})
      for name in _OPTIONAL_RUN_FILES:
        if (directory / name).exists():
          receipt = _receipt(directory / name, root, "process_wall_time")
          retained[receipt["path"]] = receipt
    pair["paired_delta_percentage_points"] = {
        key: pair["cnn_ssm"]["metrics_percent"][key]
        - pair["cnn_only"]["metrics_percent"][key] for key in METRICS}
    pairs.append(pair)
  data, data_path = _validate_data_manifest(data_manifest, runs, shared)
  data_kind = spec.get("data_kind", "unverified")
  _check(data_kind in ("real_forty", "synthetic", "unverified"),
         "Unsupported research data kind")
  if data_kind == "real_forty":
    _check(data["config"]["source_root"].rstrip("/") == GCS_ROOT
           and data["config"]["total_shards"] == TOTAL_SHARDS,
           "Real ForTy results require the official source and published shard count")
    _check(len(seeds) >= 3, "A real ForTy research pilot requires at least three seeds")
    provenance = _real_provenance(root, data, data_path, runs)
    status = "completed_pilot"
  elif data_kind == "synthetic":
    status = "software_check_complete"
    provenance = {"status": "software_check_only"}
  else:
    status = "artifacts_complete_data_kind_unverified"
    provenance = {"status": "data_origin_unverified"}
  summary = {}
  for arm in ARMS:
    summary[arm] = {
        "metrics_percent": {key: _stats([pair[arm]["metrics_percent"][key]
                                        for pair in pairs]) for key in METRICS},
        "parameter_count": pairs[0][arm]["parameter_count"],
    }
  summary["paired_delta_percentage_points"] = {
      key: _stats([pair["paired_delta_percentage_points"][key] for pair in pairs])
      for key in METRICS}
  for path in [data_path, *_immutable_receipts(root)]:
    if path.resolve() == destination:
      continue
    receipt = _receipt(path, root, "provenance_receipt")
    retained.setdefault(receipt["path"], receipt)
  retention = {
      "schema_version": 1, "manifest_kind": "architecture_pilot_checkpoint_retention",
      "status": status, "run_root": str(root),
      "selected_checkpoint_count": len(seeds) * len(ARMS),
      "checkpoints": checkpoint_records,
      "files": [retained[name] for name in sorted(retained)],
      "scope": (
          "Checkpoint retention records weights only, not optimizer state. "
          "Completed runs are validated before reuse; partial training restarts "
          "from its seed. Filesystem replacement and persistent storage do not "
          "guarantee survival of all runtime-reclamation failures. Live runner "
          "ledgers and derived reports are excluded from this immutable receipt set."),
  }
  retention_path = root / "checkpoint_manifest.json"
  if retention_path.exists():
    _check(_json(retention_path).get("manifest_kind")
           == "architecture_pilot_checkpoint_retention",
           "Existing non-retention checkpoint manifest is preserved")
  _atomic_json(retention_path, retention)
  counts = data["config"]
  report = {
      "schema_version": 1, "report_kind": "neutral_architecture_pilot",
      "status": status, "data_kind": data_kind,
      "comparison": "architecture", "seeds": seeds, "seed_count": len(seeds),
      "research_pilot_multiseed": len(seeds) > 1,
      "provenance_validation": provenance,
      "class_ids": list(range(NUM_CLASSES)),
      "label_semantics": {
          "status": "unverified", "class_names_reported": False,
          "publisher_question": (
              "https://github.com/google-deepmind/forest_typology/issues/2"),
          "unknown_protocol": (
              "No numeric ID is ignored; publisher Unknown protocol is unconfirmed")},
      "metric_conventions": {
          "macro_f1_all_ids": (
              "Macro F1 over all nine numeric IDs; absent IDs have F1 zero"),
          "mean_iou_present_ids": (
              "Mean IoU over IDs with nonzero true or predicted support"),
          "pixel_accuracy": "Correct pixels divided by all evaluated pixels",
          "units": "percent; paired differences are percentage points"},
      "shared_cohort": shared,
      "coverage": {"source_root": counts["source_root"],
                   "source_shards_per_split": counts["total_shards"],
                   "published_shards_per_split": (
                       TOTAL_SHARDS if data_kind == "real_forty" else None),
                   "train_shards": counts["train_shards"],
                   "validation_shards": counts["val_shards"],
                   "test_shards": counts["test_shards"],
                   "full_published_test_split": (
                       data_kind == "real_forty"
                       and counts["test_shards"] == TOTAL_SHARDS)},
      "pairs": pairs, "summary": summary,
      "delta_direction": "cnn_ssm_minus_cnn_only",
      "retention_manifest": {"path": retention_path.name,
                             "sha256": _digest(retention_path)},
      "limitations": [
          "A single-seed check has no sample standard deviation or seed-variance "
          "estimate."
          if len(seeds) == 1 else
          "Seed means and sample standard deviations are descriptive; no significance "
          "claim follows.",
          "CNN-only uses temporal mean pooling and omits selective SSM blocks; "
          "parameter budgets are not matched.",
          "Shared examples and training settings do not imply identical augmentation "
          "draws, optimizer updates or trajectories.",
          "Test pixels repeat across seeds; matrices are not pooled into a larger "
          "independent test set.",
          "Numeric-ID metrics do not establish semantic forest-class scores or the "
          "publisher's official Unknown protocol.",
          "Selected shards and sample checks do not establish whole-dataset validity "
          "or geographic independence.",
      ],
  }
  _atomic_json(destination, report)
  return report


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run-root", required=True, type=Path)
  parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
  parser.add_argument("--data-manifest", required=True, type=Path)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args(argv)
  report = build_report(args.run_root, args.seeds, data_manifest=args.data_manifest,
                        output=args.output)
  print(json.dumps({"status": report["status"], "summary": report["summary"]},
                   indent=2, allow_nan=False))
  return report


if __name__ == "__main__":
  main()
