#!/usr/bin/env python3
"""Audit the immutable pilot metadata and regenerate numeric-ID analysis offline.

Run with Python and matplotlib installed. TensorFlow, weights and source records
are not needed or loaded. Missing weight/record bytes remain explicitly unverified.
"""

import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "metadata"
ZIP_SHA256 = "ba644a9633e3af063dd083bf513d0b38319f9084d755ee0aa2dee9fd337bd3ba"
SOURCE = "7f55b2c30c830693873e1fe1628c8e52208ed57d"
SEEDS = (42, 43, 44)
ARMS = ("cnn_only", "cnn_ssm")
METRICS = ("macro_f1_all_ids", "mean_iou_present_ids", "pixel_accuracy")


def require(condition, message):
  if not condition:
    raise ValueError(message)


def digest(content):
  return hashlib.sha256(content).hexdigest()


def load(name):
  return json.loads((DATA / name).read_text())


def checked_path(name):
  path = PurePosixPath(name)
  require(not path.is_absolute() and ".." not in path.parts, name)
  return DATA / name


def verify_entries(entries):
  missing = []
  for entry in entries:
    require(len(entry["sha256"]) == 64 and
            set(entry["sha256"]) <= set("0123456789abcdef"), "SHA format")
    require(type(entry["bytes"]) is int and entry["bytes"] > 0, "Recorded size")
    path = checked_path(entry["path"])
    if not path.is_file():
      require(path.name.endswith(".weights.h5"), "Missing " + entry["path"])
      missing.append(entry["path"])
      continue
    content = path.read_bytes()
    require(len(content) == entry["bytes"], "Size: " + entry["path"])
    require(digest(content) == entry["sha256"], "SHA: " + entry["path"])
  return {"references": len(entries), "present_verified": len(entries) - len(missing),
          "absent_weights": len(missing), "absent_weight_paths": sorted(missing)}


def provenance():
  require(digest((ROOT / "metadata.zip").read_bytes()) == ZIP_SHA256, "ZIP SHA")
  with zipfile.ZipFile(ROOT / "metadata.zip") as archive:
    names = archive.namelist()
    require(len(names) == 96 and len(set(names)) == 96, "ZIP membership")
    actual = {str(p.relative_to(DATA)) for p in DATA.rglob("*") if p.is_file()}
    require(set(names) == actual, "Extracted metadata membership")
    for name in names:
      require(archive.read(name) == checked_path(name).read_bytes(), name)
  saved = load("profile-archives.json")["metadata"]["saved"]
  require(len(saved) == 95, "Archive receipt count")
  require({e["path"] for e in saved} == set(names) - {"profile-archives.json"},
          "Archive receipt membership")
  profiles = verify_entries(saved)
  retention = load("checkpoint_manifest.json")
  retained = verify_entries(retention["files"])
  require((retained["references"], retained["present_verified"],
           retained["absent_weights"]) == (98, 86, 12), "Retention counts")
  require(len(retention["checkpoints"]) == 12, "Checkpoint count")
  require(verify_entries(retention["checkpoints"])["absent_weights"] == 12,
          "Expected weights to remain outside metadata ZIP")
  weights = {entry["path"]: entry for entry in retention["checkpoints"]}
  inventory = load("checkpoint_inventory.json")
  require(inventory["completed_training_runs"] == 6 and
          inventory["required_training_runs"] == 6, "Checkpoint inventory")
  for run in inventory["runs"]:
    for kind in ("best", "last"):
      entry = run[kind]
      require(all(entry[k] == weights[entry["path"]][k]
                  for k in ("bytes", "sha256")), "Inventory weight receipt")
  state = load("research_state.json")
  require(state["state"] == "completed" and len(state["stages"]) == 16,
          "Incomplete execution ledger")
  outputs = []
  for stage, record in state["stages"].items():
    receipt_path = "receipts/" + stage + ".json"
    require(record["state"] == "completed", stage)
    require(digest(checked_path(receipt_path).read_bytes()) ==
            record["receipt_sha256"], "Stage receipt SHA: " + stage)
    receipt = load(receipt_path)
    require(receipt["stage_id"] == stage, "Stage identity")
    outputs.extend(receipt["outputs"])
  receipts = verify_entries(outputs)
  for entry in outputs:
    if entry["path"] in weights:
      require(all(entry[k] == weights[entry["path"]][k]
                  for k in ("bytes", "sha256")), "Stage weight receipt")
  require((receipts["present_verified"], receipts["absent_weights"]) == (46, 12),
          "Stage output counts")
  selected, identity = load("selected_data.json"), load("data_identity.json")
  require(identity["manifest_sha256"] ==
          digest((DATA / "selected_data.json").read_bytes()), "Data manifest SHA")
  require(selected["config"] == identity["config"] and
          selected["files"] == identity["files"], "Data identity contents")
  require(identity["data_kind"] == "real_forty" and
          len(selected["files"]) == 20, "Data identity scope")
  counts = {split: sum(e["split"] == split for e in selected["files"])
            for split in ("train", "validation", "test")}
  require(counts == {"train": 8, "validation": 4, "test": 8} and
          selected["config"]["total_shards"] == 1024 and
          selected["config"]["selection_seed"] == 2026, "Selected cohort budgets")
  source = load("source_identity.json")
  require(source["commit"] == SOURCE and len(source["module_sha256"]) == 13,
          "Recorded source identity")
  source_status = "unavailable_without_git_commit"
  try:
    probe = subprocess.run(["git", "show", SOURCE + ":src/mamba_forest/train.py"],
                           cwd=ROOT, capture_output=True)
  except FileNotFoundError:
    probe = None
  if probe is not None and probe.returncode == 0:
    for name, sha in source["module_sha256"].items():
      require(Path(name).name == name, "Source filename")
      content = subprocess.check_output(
          ["git", "show", SOURCE + ":src/mamba_forest/" + name], cwd=ROOT)
      require(digest(content) == sha, "Source module: " + name)
    source_status = "13_recorded_module_hashes_match_git_commit"
  return {"zip_sha256": ZIP_SHA256, "zip_entries_match_extracted_bytes": 96,
          "archive_saved_receipts": profiles, "retention_manifest": retained,
          "completed_stage_receipt_hashes_verified": 16,
          "stage_output_receipts": receipts,
          "source_commit": SOURCE, "source_module_check": source_status,
          "selected_data_manifest_sha256": identity["manifest_sha256"],
          "selected_data_receipts": 20,
          "recorded_checkpoint_files": len(weights),
          "recorded_checkpoint_total_bytes": sum(e["bytes"] for e in weights.values()),
          "recorded_unique_weight_hashes": len({e["sha256"] for e in weights.values()}),
          "record_bytes_and_weight_bytes_rehashed_locally": False}


def matrix_metrics(matrix):
  require(len(matrix) == 9 and all(len(r) == 9 for r in matrix), "Matrix shape")
  require(all(type(v) is int and v >= 0 for r in matrix for v in r),
          "Matrix must contain nonnegative integers")
  support = list(map(sum, matrix))
  predicted = [sum(row[k] for row in matrix) for k in range(9)]
  classes = []
  for k in range(9):
    denominator = support[k] + predicted[k]
    classes.append({"id": k, "support": support[k],
                    "predicted_pixels": predicted[k],
                    "f1_percent": 200 * matrix[k][k] / denominator
                    if denominator else 0.,
                    "iou_percent": 100 * matrix[k][k] /
                    (denominator - matrix[k][k]) if denominator else 0.})
  present = [c for c in classes if c["support"] + c["predicted_pixels"]]
  metrics = {"macro_f1_all_ids": statistics.mean(c["f1_percent"] for c in classes),
             "mean_iou_present_ids": statistics.mean(c["iou_percent"]
                                                     for c in present),
             "pixel_accuracy": 100 * sum(matrix[k][k] for k in range(9)) /
             sum(support)}
  return metrics, classes


def same_number(actual, expected):
  require(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10),
          "Metric mismatch: " + str((actual, expected)))


def analyze():
  report = load("research_report.json")
  require(report["status"] == "completed_pilot" and
          report["seeds"] == list(SEEDS), "Pilot status or seeds")
  config = load("research_spec.json")["config"]
  require(config["seeds"] == list(SEEDS) and config["epochs"] == 10 and
          config["selection_seed"] == 2026, "Frozen protocol")
  rows, class_rows, common_config, support_reference = [], [], None, None
  checkpoint_by_path = {e["path"]: e for e in load("checkpoint_manifest.json")
                        ["checkpoints"]}
  for seed in SEEDS:
    pair = next(p for p in report["pairs"] if p["seed"] == seed)
    for arm in ARMS:
      prefix = arm + "_" + str(seed)
      test = load(prefix + "/test/test_metrics.json")
      cfg = load(prefix + "/config.json")
      metrics, classes = matrix_metrics(test["confusion_matrix"])
      require(test["confusion_matrix"] == pair[arm]["confusion_matrix"],
              "Report matrix mismatch")
      require(test["parameters"] == pair[arm]["parameter_count"] ==
              report["summary"][arm]["parameter_count"] ==
              {"cnn_only": 5035392, "cnn_ssm": 7732800}[arm], "Parameter count")
      for metric in METRICS:
        same_number(metrics[metric], pair[arm]["metrics_percent"][metric])
      for raw, metric in zip(("overall_f1", "mean_iou", "pixel_accuracy"), METRICS):
        same_number(test[raw] * 100, metrics[metric])
      for actual, expected in zip(classes, pair[arm]["per_numeric_class"]):
        for key in ("support", "predicted_pixels", "f1_percent", "iou_percent"):
          same_number(actual[key], expected[key])
        class_rows.append({"seed": seed, "model": arm, **actual})
      supports = [c["support"] for c in classes]
      if support_reference is None:
        support_reference = supports
      require(supports == support_reference and sum(supports) == test["num_pixels"],
              "Test cohort support")
      controls = {k: v for k, v in cfg.items()
                  if k not in ("output_dir", "seed", "model_variant")}
      if common_config is None:
        common_config = controls
      require(controls == common_config and cfg["seed"] == seed and
              cfg["model_variant"] == arm and cfg["fixed_train_subset"],
              "Training controls differ")
      with (DATA / prefix / "training_results.csv").open(newline="") as handle:
        history = list(csv.DictReader(handle))
      split = load(prefix + "/split_manifest.json")
      require(len(history) == 10 and len(split["epochs"]) == 10, "Epoch count")
      require(split["selection_seed"] == 2026 and
              split["validation_paths"] == report["shared_cohort"]["paths"]
              ["validation"], "Validation cohort")
      for epoch, (csv_row, split_row) in enumerate(zip(history, split["epochs"]), 1):
        require(int(csv_row["epoch"]) == epoch and split_row["training_completed"]
                and split_row["paths"] == report["shared_cohort"]["paths"]["train"],
                "Training cohort or completion")
        for key in ("train_batches", "optimizer_updates", "skipped_optimizer_updates"):
          require(int(csv_row[key]) == split_row[key], "Epoch update receipt")
        require(int(csv_row["optimizer_updates"]) > 0 and
                int(csv_row["train_batches"]) == int(csv_row["optimizer_updates"]) +
                int(csv_row["skipped_optimizer_updates"]), "Update accounting")
      selection = load(prefix + "/checkpoint_selection.json")
      best = max(history, key=lambda r: float(r["overall_f1"]))
      require(selection["epoch"] == int(best["epoch"]), "Selection epoch")
      same_number(selection["score"], float(best["overall_f1"]))
      require(selection["weights_sha256"] == test["weights_sha256"] ==
              checkpoint_by_path[prefix + "/best.weights.h5"]["sha256"],
              "Recorded best-checkpoint identity")
      require(test["shard_paths"] == report["shared_cohort"]["paths"]["test"],
              "Test shard cohort")
      counts = {k: sum(int(r[k]) for r in history) for k in
                ("train_examples", "train_batches", "optimizer_updates",
                 "skipped_optimizer_updates")}
      require(counts["train_examples"] == 11950 and counts["train_batches"] == 1500,
              "Training example budget")
      counts["train_example_presentations"] = counts.pop("train_examples")
      wall = {kind: float((DATA / prefix / (kind + "_wall_seconds.txt")).read_text())
              for kind in ("train", "test")}
      rows.append({"seed": seed, "model": arm, **metrics,
                   "parameters": test["parameters"], "best_epoch": selection["epoch"],
                   "best_validation_macro_f1_percent": selection["score"] * 100,
                   "test_pixels": sum(supports), **counts,
                   "train_process_wall_seconds": wall["train"],
                   "test_process_wall_seconds": wall["test"]})
  summary = {}
  for arm in (*ARMS, "paired_delta_percentage_points"):
    summary[arm] = {}
    for metric in METRICS:
      values = [next(r[metric] for r in rows if r["seed"] == s and
                     r["model"] == arm) for s in SEEDS] if arm in ARMS else [
          next(r[metric] for r in rows if r["seed"] == s and r["model"] == "cnn_ssm") -
          next(r[metric] for r in rows if r["seed"] == s and r["model"] == "cnn_only")
          for s in SEEDS]
      result = {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values),
                "seed_values": values}
      expected = report["summary"][arm]
      expected = expected["metrics_percent"] if arm in ARMS else expected
      for statistic in ("mean", "sample_sd"):
        same_number(result[statistic], expected[metric][statistic])
      summary[arm][metric] = result
  return rows, class_rows, summary


def write_csv(name, rows):
  with (ROOT / name).open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def plot(rows):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  plt.rcParams.update({"svg.hashsalt": "forty-pilot-20261029", "font.size": 10,
                       "axes.spines.top": False, "axes.spines.right": False})
  figure, axes = plt.subplots(1, 3, figsize=(11.6, 4.4))
  titles = ("Macro F1 · all 9 IDs", "mIoU · supported IDs", "Pixel accuracy")
  for axis, metric, title in zip(axes, METRICS, titles):
    for seed, color in zip(SEEDS, ("#0072B2", "#D55E00", "#009E73")):
      values = [next(r[metric] for r in rows if r["seed"] == seed and
                     r["model"] == arm) for arm in ARMS]
      axis.plot((0, 1), values, "o-", color=color, linewidth=1.8,
                markersize=6, label="Seed " + str(seed))
    axis.set(title=title, ylabel="Percent", xticks=(0, 1),
             xticklabels=("CNN-only", "CNN+SSM"), xlim=(-.22, 1.22))
    axis.grid(axis="y", alpha=.2)
  axes[0].legend(frameon=False, loc="best")
  figure.suptitle("ForTy architecture pilot · paired results across three seeds",
                  fontsize=14, fontweight="bold")
  figure.text(.5, .035,
              "One fixed cohort · 146 test tiles · 10 epochs · batch 8 · "
              "numeric IDs 0–8; semantic mapping unconfirmed\n"
              "Lines connect matching seeds. CNN+SSM uses 53.57% more parameters. "
              "These descriptive results do not establish significance.",
              ha="center", fontsize=8.7, color="#444444")
  figure.tight_layout(rect=(0, .13, 1, .92))
  figure.savefig(ROOT / "paired-seed-results.svg", metadata={"Date": None})
  svg = ROOT / "paired-seed-results.svg"
  svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) +
                 "\n")
  figure.savefig(ROOT / "paired-seed-results.png", dpi=200)
  plt.close(figure)


def main():
  checks = provenance()
  rows, class_rows, summary = analyze()
  require(all(r["predicted_pixels"] == 0 and r["support"] == 9770
              for r in class_rows if r["id"] == 7),
          "ID 7 failure diagnosis changed")
  audit = {"schema_version": 1,
           "status": "metadata_checks_passed_weights_and_records_not_present",
           "analysis_script_sha256": digest(Path(__file__).read_bytes()),
           "verification": checks, "seeds": list(SEEDS), "models": rows,
           "summary_percent_and_paired_percentage_points": summary,
           "parameters": {"cnn_only": 5035392, "cnn_ssm": 7732800,
                          "increase_percent": 100 * (7732800 / 5035392 - 1)},
           "numeric_id_7": {"support_per_test": 9770,
                            "predicted_pixels_each_of_six_runs": 0,
                            "f1_percent_each_of_six_runs": 0},
           "recorded_runtime": load("runtime_identity.json"),
           "verification_boundary": [
               "Metadata bytes and internal hash receipts were checked locally.",
               "The 12 H5 weights and 20 TFRecord shards are excluded; their "
               "recorded hashes are cross-checked, not rehashed from actual bytes.",
               "Source module hashes match the pinned commit when Git is available; "
               "metadata is not independent proof of remote execution or "
               "storage durability.",
               "Numeric IDs have unconfirmed semantics; no ID is ignored.",
               "Three seed differences are descriptive, not a significance claim; "
               "the same test pixels repeat across seeds.",
               "Process wall times include setup, data, evaluation and checkpoint "
               "I/O; they are not isolated compute performance measurements."]}
  (ROOT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
  write_csv("paired-seed-results.csv", rows)
  write_csv("class-metrics.csv", class_rows)
  plot(rows)
  print("Audited 96 ZIP entries, 95 saved-file receipts, 16 stage receipts; "
        "recomputed six numeric-ID matrices. Weight/record bytes are absent.")


if __name__ == "__main__":
  main()
