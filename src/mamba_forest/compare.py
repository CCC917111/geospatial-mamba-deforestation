"""Validate and summarize paired ComboLoss/plain-cross-entropy experiments.

Each seed requires complete runs in ``RUN_ROOT/combo_SEED`` and
``RUN_ROOT/plain_ce_SEED``, including test/test_metrics.json and the checkpoint
selection record. This command reads local artifacts only and imports no ML
framework. Scores are percentages; paired deltas are ComboLoss minus plain CE
in percentage points. One seed has no sample standard deviation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics

from mamba_forest.checkpoint import weights_sha256
from mamba_forest.constants import CLASS_NAMES, FOREST_CLASS_INDICES, NUM_CLASSES

SCORE_KEYS = ("overall_f1", "forests_f1", "natural_forest_f1",
              "planted_forest_f1", "tree_crops_f1")
_OBJECTIVE_KEYS = {"combo_alpha", "label_smoothing", "no_class_weights"}
_MODEL_KEYS = ("d_model", "d_state_temporal", "d_state_spatial", "bottleneck_channels")
_CONVENTIONS = {
    "overall_f1": "macro mean over all classes; absent classes have F1 0",
    "mean_iou": "mean over classes with nonzero true or predicted support",
}


def _check(condition, message):
  if not condition:
    raise ValueError(message)


def _number(value, label, minimum=None, maximum=None):
  _check(isinstance(value, (int, float)) and not isinstance(value, bool)
         and math.isfinite(value), f"{label} must be finite and numeric")
  _check(minimum is None or value >= minimum, f"{label} is below {minimum}")
  _check(maximum is None or value <= maximum, f"{label} exceeds {maximum}")
  return value


def _integer(value, label, minimum=0):
  _number(value, label, minimum)
  _check(int(value) == value, f"{label} must be an integer")
  return int(value)


def _read_json(path):
  def reject_constant(value):
    raise ValueError(f"non-finite JSON constant {value}")

  with path.open(encoding="utf-8") as handle:
    result = json.load(handle, parse_constant=reject_constant)
  _check(isinstance(result, dict), f"{path} must contain a JSON object")
  return result


def _wall_seconds(path):
  if not path.exists():
    return None
  value = _number(float(path.read_text(encoding="utf-8").strip()), str(path), 0)
  _check(value > 0, f"{path}: wall time must be positive")
  return value


def _paths(config, split, count):
  total = _integer(config["total_shards"], "total_shards", 1)
  count = _integer(count, f"{split} shard count", 1)
  _check(count <= total, f"{split} shard count exceeds total_shards")
  paths = [f"{config['gcs_root']}/forty_v1-{split}.tfrecord-{index:05d}"
           f"-of-{total:05d}" for index in range(total)]
  return (random.Random(config["seed"]).sample(paths, count)
          if split == "train" else paths[:count])


def _read_history(path, epochs):
  required = {"epoch", "train_loss", "val_accuracy", "val_mean_iou",
              "overall_f1", "forests_f1", "train_seconds", "validation_seconds",
              "train_examples", "train_examples_per_second"}
  required.update("f1_" + name.lower().replace(" ", "_") for name in CLASS_NAMES)
  with path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    _check(required <= set(reader.fieldnames or []), f"{path}: missing CSV columns")
    rows = []
    for row in reader:
      converted = {key: _number(float(value), f"{path}: {key}")
                   for key, value in row.items()}
      for key, value in converted.items():
        bounded = key in {"val_accuracy", "val_mean_iou", "overall_f1", "forests_f1"}
        if bounded or key.startswith("f1_"):
          _number(value, key, 0, 1)
      f1 = [converted["f1_" + name.lower().replace(" ", "_")]
            for name in CLASS_NAMES]
      aggregates = {
          "overall_f1": statistics.mean(f1),
          "forests_f1": statistics.mean(f1[index] for index in FOREST_CLASS_INDICES),
      }
      for key, expected in aggregates.items():
        _check(math.isclose(converted[key], expected, rel_tol=1e-9, abs_tol=1e-12),
               f"{path}: {key} differs from per-class F1")
      for key in ("train_seconds", "validation_seconds", "train_examples_per_second"):
        _check(converted[key] > 0, f"{path}: {key} must be positive")
      converted["train_examples"] = _integer(converted["train_examples"],
                                             "train_examples", 1)
      rows.append(converted)
  _check([row["epoch"] for row in rows] == list(range(1, epochs + 1)),
         f"{path}: training must complete every configured epoch")
  return rows


def _validate_metrics(metrics, config, checkpoint):
  _check(metrics["split"] == "test", "evaluation must use the test split")
  _check(metrics["gcs_root"] == config["gcs_root"],
         "test data root differs from training")
  _check(metrics["total_shards_per_split"] == config["total_shards"],
         "test total shard count differs from training")
  _check(metrics["shard_paths"] == _paths(config, "test", metrics["test_shards"]),
         "test shard paths do not match the fixed selection")
  _integer(metrics["batch_size"], "test batch_size", 1)
  _check(metrics["weights_sha256"] == weights_sha256(checkpoint),
         "test checkpoint SHA256 does not match best.weights.h5")
  _check(metrics["confusion_axes"] == {"rows": "true", "columns": "predicted"},
         "unsupported confusion matrix axes")
  _check(metrics["metric_conventions"] == _CONVENTIONS,
         "unsupported metric conventions")
  matrix = metrics["confusion_matrix"]
  _check(isinstance(matrix, list) and len(matrix) == NUM_CLASSES,
         "confusion matrix must have one row per class")
  for row in matrix:
    _check(isinstance(row, list) and len(row) == NUM_CLASSES,
           "confusion matrix must be square")
    for value in row:
      _integer(value, "confusion count")
  support = [sum(row) for row in matrix]
  predicted = [sum(row[index] for row in matrix) for index in range(NUM_CLASSES)]
  total = sum(support)
  _integer(metrics["num_pixels"], "num_pixels", 1)
  for value in metrics["class_support"].values():
    _integer(value, "class support count")
  _check(total > 0 and metrics["num_pixels"] == total,
         "confusion total/num_pixels mismatch")
  _check(metrics["class_support"] == dict(zip(CLASS_NAMES, support)),
         "class_support differs from confusion matrix")
  f1 = [2 * matrix[index][index] / (support[index] + predicted[index])
        if support[index] + predicted[index] else 0.0 for index in range(NUM_CLASSES)]
  union = [support[index] + predicted[index] - matrix[index][index]
           for index in range(NUM_CLASSES)]
  iou = [matrix[index][index] / union[index]
         for index in range(NUM_CLASSES) if union[index] > 0]
  expected = {
      "pixel_accuracy": sum(matrix[index][index]
                            for index in range(NUM_CLASSES)) / total,
      "mean_iou": statistics.mean(iou), "overall_f1": statistics.mean(f1),
      "forests_f1": statistics.mean(f1[index] for index in FOREST_CLASS_INDICES),
      "natural_forest_f1": f1[FOREST_CLASS_INDICES[0]],
      "planted_forest_f1": f1[FOREST_CLASS_INDICES[1]],
      "tree_crops_f1": f1[FOREST_CLASS_INDICES[2]],
  }
  for key, value in expected.items():
    actual = _number(metrics[key], key, 0, 1)
    _check(math.isclose(actual, value, rel_tol=1e-9, abs_tol=1e-12),
           f"{key} differs from confusion matrix")
  _check(set(metrics["per_class_f1"]) == set(CLASS_NAMES),
         "per-class F1 names mismatch")
  for name, value in zip(CLASS_NAMES, f1):
    actual = _number(metrics["per_class_f1"][name], f"F1 {name}", 0, 1)
    _check(math.isclose(actual, value, rel_tol=1e-9, abs_tol=1e-12),
           f"F1 {name} differs from confusion matrix")
  model = metrics["model_config"]
  _check(model["num_classes"] == NUM_CLASSES, "model class count mismatch")
  for key in _MODEL_KEYS:
    _check(model[key] == config[key], f"test model {key} differs from training")
  _check(model["dropout_rate"] == config["dropout"],
         "test model dropout differs from training")
  for key in ("tensorflow_version", "precision_policy"):
    _check(isinstance(metrics["runtime"][key], str) and metrics["runtime"][key],
           f"test runtime {key} is missing")


def _validate_config(config, variant, seed):
  required = _OBJECTIVE_KEYS | set(_MODEL_KEYS) | {
      "seed", "gcs_root", "total_shards", "train_shards", "val_shards", "batch_size",
      "epochs", "fixed_train_subset", "learning_rate", "lr_alpha", "weight_decay",
      "dropout", "grad_clip", "patience", "precision", "no_mixed_precision",
      "resolved_precision", "tensorflow_version",
  }
  _check(required <= set(config),
         f"missing training condition keys: {sorted(required - set(config))}")
  _check(variant in ("combo", "plain_ce"), "unknown objective variant")
  expected = (0.4, 0.05, False) if variant == "combo" else (1.0, 0.0, True)
  actual = tuple(config[key] for key in
                 ("combo_alpha", "label_smoothing", "no_class_weights"))
  _check(isinstance(config["no_class_weights"], bool) and actual == expected,
         f"{variant} objective configuration is incorrect")
  _check(_integer(config["seed"], "seed") == seed,
         "run seed differs from requested seed")
  _check(config["fixed_train_subset"] is True,
         "comparison requires fixed training subsets")
  for key in (*_MODEL_KEYS, "total_shards", "train_shards", "val_shards",
              "batch_size", "epochs"):
    _integer(config[key], key, 1)
  _integer(config["patience"], "patience")
  for key in ("learning_rate", "weight_decay", "grad_clip"):
    _number(config[key], key, 0)
  _number(config["lr_alpha"], "lr_alpha", 0, 1)
  _number(config["dropout"], "dropout", 0, 1)
  _check(config["dropout"] < 1, "dropout must be less than 1")
  _check(isinstance(config["gcs_root"], str) and config["gcs_root"],
         "data root is missing")
  _check(config["resolved_precision"] in ("float32", "mixed_float16"),
         "training resolved_precision is invalid")
  _check(config["precision"] in ("auto", "float32", "mixed_float16")
         and isinstance(config["no_mixed_precision"], bool),
         "training precision is invalid")
  explicit = "float32" if config["no_mixed_precision"] else config["precision"]
  _check(explicit == "auto" or explicit == config["resolved_precision"],
         "training precision disagrees with resolved_precision")
  _check(isinstance(config["tensorflow_version"], str) and config["tensorflow_version"],
         "training tensorflow_version is missing")


def validate_training_run(directory, variant, seed) -> dict:
  """Validates complete training artifacts before any test evaluation is opened."""
  directory = Path(directory)
  try:
    config = _read_json(directory / "config.json")
    manifest = _read_json(directory / "split_manifest.json")
    selection = _read_json(directory / "checkpoint_selection.json")
    _validate_config(config, variant, seed)
    epochs = _integer(config["epochs"], "epochs", 1)
    rows = _read_history(directory / "training_results.csv", epochs)
    train_paths = _paths(config, "train", config["train_shards"])
    _check(manifest["schema_version"] == 1 and manifest["selection_protocol"]
           == "python-random-seed-plus-epoch-v1", "unsupported split manifest protocol")
    _check(manifest["gcs_root"] == config["gcs_root"]
           and manifest["total_shards_per_split"] == config["total_shards"]
           and manifest["seed"] == seed and manifest["training_mode"] == "fixed",
           "manifest configuration differs from training")
    _check(manifest["validation_paths"]
           == _paths(config, "validation", config["val_shards"]),
           "validation paths differ from fixed selection")
    _check(len(manifest["epochs"]) == epochs, "manifest epoch count is incomplete")
    for index, entry in enumerate(manifest["epochs"], 1):
      _check(entry["epoch"] == index and entry["shard_epoch"] == 0
             and entry["training_completed"] is True and entry["paths"] == train_paths,
             "manifest training epochs are incomplete or do not reuse the fixed subset")
    _check(manifest["completed_unique_train_shards"] == len(train_paths),
           "completed unique training coverage is incorrect")
    best = max(rows, key=lambda row: row["overall_f1"])
    _check(selection["schema_version"] == 1
           and selection["criterion"] == "validation_overall_f1"
           and selection["mode"] == "max" and selection["tie_break"] == "first"
           and selection["weights"] == "best.weights.h5"
           and selection["epoch"] == best["epoch"],
           "checkpoint selection metadata mismatch")
    _check(math.isclose(_number(selection["score"], "selected score", 0, 1),
                        best["overall_f1"], rel_tol=1e-15, abs_tol=1e-15),
           "checkpoint selection score differs from the first best validation epoch")
    _check((directory / "best.weights.h5").is_file(), "best.weights.h5 is missing")
    _check(selection["weights_sha256"] == weights_sha256(directory / "best.weights.h5"),
           "training-selected checkpoint SHA256 does not match best.weights.h5")
    return {"config": config, "manifest": manifest, "rows": rows, "best": best}
  except (ValueError, KeyError, TypeError, IndexError) as error:
    raise ValueError(f"{directory}: {error}") from error


def _read_run(directory, variant, seed):
  training = validate_training_run(directory, variant, seed)
  config, rows, best = training["config"], training["rows"], training["best"]
  try:
    metrics = _read_json(directory / "test" / "test_metrics.json")
    _validate_metrics(metrics, config, directory / "best.weights.h5")
    train_seconds = sum(row["train_seconds"] for row in rows)
    train_examples = sum(row["train_examples"] for row in rows)
    later_seconds = sum(row["train_seconds"] for row in rows[1:])
    later_examples = sum(row["train_examples"] for row in rows[1:])
    return {**training, "metrics": metrics,
            "summary": {
                "run_dir": str(directory), "epochs_completed": len(rows),
                "best_validation_epoch": int(best["epoch"]),
                "best_validation_overall_f1": best["overall_f1"],
                "scores_percent": {key: 100 * metrics[key] for key in SCORE_KEYS},
                "train_seconds": train_seconds,
                "validation_seconds": sum(row["validation_seconds"] for row in rows),
                "train_examples": train_examples,
                "train_examples_per_second": train_examples / train_seconds,
                "train_wall_seconds": _wall_seconds(
                    directory / "train_wall_seconds.txt"),
                "test_wall_seconds": _wall_seconds(directory / "test_wall_seconds.txt"),
                "epoch_timing": {
                    "first_epoch": {key: rows[0][key] for key in (
                        "train_seconds", "validation_seconds", "train_examples")},
                    "later_epochs": {
                        "count": len(rows) - 1, "train_seconds": later_seconds,
                        "validation_seconds": sum(
                            row["validation_seconds"] for row in rows[1:]),
                        "train_examples": later_examples,
                        "train_examples_per_second": (later_examples / later_seconds
                                                      if later_seconds else None),
                    },
                },
            }}
  except (ValueError, KeyError, TypeError, IndexError) as error:
    raise ValueError(f"{directory}: {error}") from error


def _effective_config(config):
  ignored = _OBJECTIVE_KEYS | {"output_dir", "precision", "no_mixed_precision"}
  return {key: value for key, value in config.items() if key not in ignored}


def _test_protocol(metrics):
  return {
      **{key: metrics[key] for key in (
          "gcs_root", "total_shards_per_split", "test_shards", "shard_paths",
          "batch_size", "class_support", "num_pixels", "metric_conventions",
          "confusion_axes")},
      "model_config": {key: value for key, value in metrics["model_config"].items()
                       if key != "name"},
      "runtime": {key: metrics["runtime"][key]
                  for key in ("tensorflow_version", "precision_policy")},
  }


def _stats(values):
  return {"mean": statistics.mean(values),
          "sample_sd": statistics.stdev(values) if len(values) > 1 else None}


def summarize_runs(run_root, seeds=(42, 43, 44)) -> dict:
  """Rejects unmatched/incomplete local artifacts and summarizes paired scores."""
  seeds = list(seeds)
  _check(seeds and len(set(seeds)) == len(seeds), "provide at least one unique seed")
  for seed in seeds:
    _integer(seed, "seed", 0)
  root = Path(run_root)
  pairs, reference_config, reference_protocol = [], None, None
  for seed in seeds:
    combo = _read_run(root / f"combo_{seed}", "combo", seed)
    plain = _read_run(root / f"plain_ce_{seed}", "plain_ce", seed)
    config = _effective_config(combo["config"])
    protocol = _test_protocol(combo["metrics"])
    _check(config == _effective_config(plain["config"]),
           f"seed {seed}: training config mismatch")
    _check(combo["manifest"] == plain["manifest"],
           f"seed {seed}: split manifests mismatch")
    _check(protocol == _test_protocol(plain["metrics"]),
           f"seed {seed}: test protocol mismatch")
    _check([row["train_examples"] for row in combo["rows"]]
           == [row["train_examples"] for row in plain["rows"]],
           f"seed {seed}: training example budgets mismatch")
    across_seeds = {key: value for key, value in config.items() if key != "seed"}
    if reference_config is None:
      reference_config, reference_protocol = across_seeds, protocol
    _check(across_seeds == reference_config, "training conditions differ across seeds")
    _check(protocol == reference_protocol, "test conditions differ across seeds")
    pairs.append({
        "seed": seed, "combo": combo["summary"], "plain_ce": plain["summary"],
        "paired_delta_percentage_points": {
            key: 100 * (combo["metrics"][key] - plain["metrics"][key])
            for key in SCORE_KEYS},
    })
  summary = {}
  for variant in ("combo", "plain_ce"):
    summary[variant] = {
        "scores_percent": {key: _stats([pair[variant]["scores_percent"][key]
                                       for pair in pairs])
                           for key in SCORE_KEYS},
        **{key: _stats([pair[variant][key] for pair in pairs])
           for key in ("train_seconds", "validation_seconds", "train_examples")},
        "train_examples_per_second": (
            sum(pair[variant]["train_examples"] for pair in pairs)
            / sum(pair[variant]["train_seconds"] for pair in pairs)),
        **{key: (_stats([pair[variant][key] for pair in pairs])
                 if all(pair[variant][key] is not None for pair in pairs) else None)
           for key in ("train_wall_seconds", "test_wall_seconds")},
    }
  summary["paired_delta_percentage_points"] = {
      key: _stats([pair["paired_delta_percentage_points"][key] for pair in pairs])
      for key in SCORE_KEYS}
  return {"schema_version": 1, "run_root": str(root), "seeds": seeds,
          "delta_direction": "combo_minus_plain_ce", "pairs": pairs, "summary": summary,
          "test_coverage": {
              "shards": reference_protocol["test_shards"],
              "total_shards": reference_protocol["total_shards_per_split"],
              "full_split": (reference_protocol["test_shards"]
                             == reference_protocol["total_shards_per_split"]),
          },
          "timing_scope": "Reported epoch training/validation time; excludes "
                          "model build and checkpoint I/O."}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run-root", required=True)
  parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
  parser.add_argument("--output", help="Defaults to RUN_ROOT/comparison.json")
  args = parser.parse_args()
  try:
    report = summarize_runs(args.run_root, args.seeds)
    output = (Path(args.output) if args.output
              else Path(args.run_root) / "comparison.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(output)
  except (ValueError, OSError, csv.Error) as error:
    parser.error(str(error))
  print(f"paired comparison written to {output}")


if __name__ == "__main__":
  main()
