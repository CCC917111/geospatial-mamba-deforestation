"""Plan or execute paired, fixed-split objective or architecture experiments.

Planning writes argv lists without importing TensorFlow or opening data. Repeat
the same arguments with --execute to run that reviewed plan. Existing execution
artifacts are never reused; select a fresh root after an interrupted or failed
run. --allow-cpu supports small local pipeline checks, not GPU validation.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from mamba_forest.compare import COMPARISON_ARMS, validate_training_run
from mamba_forest.constants import GCS_ROOT, MODEL_DEFAULTS, TOTAL_SHARDS

ARMS = ("combo", "plain_ce")
_MODEL_KEYS = {"dropout": "dropout_rate", **{
    name: name for name in MODEL_DEFAULTS if name != "dropout_rate"}}
_RUNTIME_PROBE = """import json, sys
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
required = sys.argv[1] == 'gpu'
if required and not gpus:
    raise RuntimeError('GPU required; use --allow-cpu only for local checks')
tf.config.set_soft_device_placement(False)
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    product = tf.matmul(tf.ones((2, 2)), tf.ones((2, 2)))
product.numpy()
if required and 'GPU:0' not in product.device:
    raise RuntimeError('GPU probe silently fell back to CPU')
print(json.dumps({'tensorflow_version': tf.__version__,
                  'devices': [d.name for d in tf.config.list_physical_devices()],
                  'probe_device': product.device}))
"""


def parse_args(argv=None):
  """Parses experiment settings without framework imports or dataset access."""
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--run-root", required=True, type=Path)
  parser.add_argument("--comparison", choices=tuple(COMPARISON_ARMS),
                      default="objective",
                      help="objective: ComboLoss/plain CE; "
                           "architecture: CNN+SSM/CNN-only")
  parser.add_argument("--execute", action="store_true")
  parser.add_argument("--allow-cpu", action="store_true")
  parser.add_argument("--data-root", "--gcs-root", dest="data_root", default=GCS_ROOT)
  parser.add_argument("--total-shards", type=int, default=TOTAL_SHARDS)
  parser.add_argument("--train-shards", type=int, default=128)
  parser.add_argument("--val-shards", type=int, default=8)
  parser.add_argument("--test-shards", type=int,
                      help="Defaults to the full split (--total-shards)")
  parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
  parser.add_argument("--selection-seed", type=int,
                      help="Independent shared training-shard seed; defaults per run")
  parser.add_argument("--epochs", type=int, default=40)
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--precision", choices=("auto", "float32", "mixed_float16"),
                      default="auto",
                      help="auto: float32 with --allow-cpu; otherwise mixed_float16")
  for name, key in _MODEL_KEYS.items():
    parser.add_argument("--" + name.replace("_", "-"), default=MODEL_DEFAULTS[key],
                        type=float if name == "dropout" else int)
  args = parser.parse_args(argv)
  args.test_shards = args.total_shards if args.test_shards is None else args.test_shards
  capacities = [key for key in _MODEL_KEYS if key != "dropout"]
  for name in ("total_shards", "train_shards", "val_shards", "test_shards",
               "epochs", "batch_size", *capacities):
    if getattr(args, name) < 1:
      parser.error(f"--{name.replace('_', '-')} must be positive")
  budgets = ("train_shards", "val_shards", "test_shards")
  if any(getattr(args, name) > args.total_shards for name in budgets):
    parser.error("split shard budgets cannot exceed --total-shards")
  if len(set(args.seeds)) != len(args.seeds):
    parser.error("--seeds must be unique")
  if any(seed < 0 or seed > 2**32 - 1 for seed in args.seeds):
    parser.error("--seeds must be in [0, 2**32 - 1]")
  if args.selection_seed is not None and not 0 <= args.selection_seed <= 2**32 - 1:
    parser.error("--selection-seed must be in [0, 2**32 - 1]")
  if not math.isfinite(args.dropout) or not 0 <= args.dropout < 1:
    parser.error("--dropout must be finite and in [0, 1)")
  if not args.data_root:
    parser.error("--data-root must not be empty")
  return args


def make_plan(args):
  """Returns deterministic argv lists; does not inspect the dataset or runtime."""
  root = args.run_root.expanduser().resolve()
  data_root = args.data_root if "://" in args.data_root else str(
      Path(args.data_root).expanduser().resolve())
  precision = args.precision
  if precision == "auto":
    precision = "float32" if args.allow_cpu else "mixed_float16"
  config = {
      "comparison": args.comparison,
      "run_root": str(root), "data_root": data_root,
      "total_shards": args.total_shards, "train_shards": args.train_shards,
      "val_shards": args.val_shards, "test_shards": args.test_shards,
      "seeds": args.seeds, "epochs": args.epochs, "batch_size": args.batch_size,
      "selection_seed": args.selection_seed,
      "precision": precision, "evaluation_precision": "float32",
      "allow_cpu": args.allow_cpu,
      **{key: getattr(args, key) for key in _MODEL_KEYS},
  }
  python = sys.executable
  arms = COMPARISON_ARMS[args.comparison]
  model_flags = [value for key in _MODEL_KEYS for value in (
      "--" + key.replace("_", "-"), str(config[key]))]
  source_flags = ["--data-root", data_root, "--total-shards", str(args.total_shards)]
  selection_flags = ([] if args.selection_seed is None else
                     ["--selection-seed", str(args.selection_seed)])
  steps = [{
      "id": "runtime_probe", "phase": "probe",
      "argv": [python, "-c", _RUNTIME_PROBE, "cpu" if args.allow_cpu else "gpu"],
  }]
  for seed in args.seeds:
    steps.append({
        "id": f"preflight_{seed}", "phase": "preflight",
        "argv": [python, "-m", "mamba_forest.validate_data", *source_flags,
                 "--train-shards", str(args.train_shards), "--val-shards",
                 str(args.val_shards), "--test-shards", str(args.test_shards),
                 "--seed", str(seed), *selection_flags,
                 "--output", str(root / f"preflight_{seed}.json"),
                 *([] if args.allow_cpu else ["--require-gpu"])],
        "artifact": str(root / f"preflight_{seed}.json"),
    })
    for arm in arms:
      run_dir = root / f"{arm}_{seed}"
      objective = (["--combo-alpha", "0.4", "--label-smoothing", "0.05"]
                   if arm != "plain_ce" else [
                       "--combo-alpha", "1", "--no-class-weights",
                       "--label-smoothing", "0"])
      steps.append({
          "id": f"train_{arm}_{seed}", "phase": "train", "seed": seed, "arm": arm,
          "argv": [python, "-m", "mamba_forest.train", *source_flags, *model_flags,
                   "--seed", str(seed), *selection_flags,
                   "--fixed-train-subset", "--train-shards",
                   str(args.train_shards), "--val-shards", str(args.val_shards),
                   "--batch-size", str(args.batch_size), "--epochs", str(args.epochs),
                   "--patience", "0", "--precision", precision, *objective,
                   "--model-variant", arm if args.comparison == "architecture"
                   else "cnn_ssm",
                   "--output-dir", str(run_dir)],
          "wall_seconds_path": str(run_dir / "train_wall_seconds.txt"),
      })
  for seed in args.seeds:
    for arm in arms:
      run_dir = root / f"{arm}_{seed}"
      steps.append({
          "id": f"evaluate_{arm}_{seed}", "phase": "evaluate",
          "argv": [python, "-m", "mamba_forest.evaluate", *source_flags,
                   "--weights", str(run_dir / "best.weights.h5"),
                   "--test-shards", str(args.test_shards), "--batch-size",
                   str(args.batch_size), "--output-dir", str(run_dir / "test")],
          "artifact": str(run_dir / "test" / "test_metrics.json"),
          "wall_seconds_path": str(run_dir / "test_wall_seconds.txt"),
      })
  steps.append({
      "id": "compare", "phase": "compare",
      "argv": [python, "-m", "mamba_forest.compare", "--run-root", str(root),
               "--comparison", args.comparison,
               "--seeds", *[str(seed) for seed in args.seeds],
               "--output", str(root / "comparison.json")],
      "artifact": str(root / "comparison.json"),
  })
  for step in steps:
    step["log_path"] = str(root / "logs" / (step["id"] + ".log"))
  return {
      "schema_version": 1, "config": config, "steps": steps,
      "test_coverage": "full" if args.test_shards == args.total_shards else "subset",
      "scope": (f"Paired {args.comparison} comparison; preflight inspects only "
                "record 0 of one selected shard per split. CPU checks and reduced "
                "splits do not establish full-benchmark or GPU results."),
      "before_evaluation": "Verify all training artifacts and paired fixed memberships",
  }


def _atomic_json(path, payload):
  path = Path(path)
  temporary = None
  try:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8",
                                     delete=False, prefix=".json-") as handle:
      temporary = Path(handle.name)
      json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
      handle.write("\n")
    os.replace(temporary, path)
  finally:
    if temporary is not None:
      temporary.unlink(missing_ok=True)


def prepare_plan(plan):
  """Writes a fresh plan or accepts an identical, unexecuted existing plan."""
  root = Path(plan["config"]["run_root"])
  root.mkdir(parents=True, exist_ok=True)
  entries = {path.name for path in root.iterdir()}
  if entries - {"plan.json"}:
    raise ValueError(f"Run root {root} is not fresh; choose a new --run-root")
  path = root / "plan.json"
  if path.exists():
    if json.loads(path.read_text(encoding="utf-8")) != plan:
      raise ValueError(
          "Existing plan differs; repeat its arguments or choose a fresh root")
  else:
    _atomic_json(path, plan)
  return path


def _read_object(path):
  value = json.loads(Path(path).read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"Expected a JSON object: {path}")
  return value


def _verify_run(config, seed, arm):
  directory = Path(config["run_root"]) / f"{arm}_{seed}"
  for name in ("best.weights.h5", "last.weights.h5", "training_results.csv"):
    path = directory / name
    if not path.is_file() or path.stat().st_size == 0:
      raise ValueError(f"Missing or empty training artifact {path}")
  training = validate_training_run(directory, arm, seed)
  observed = training["config"]
  comparison = config.get("comparison", "objective")
  expected = {
      "gcs_root": config["data_root"], "total_shards": config["total_shards"],
      "train_shards": config["train_shards"], "val_shards": config["val_shards"],
      "batch_size": config["batch_size"], "epochs": config["epochs"],
      "seed": seed, "patience": 0, "fixed_train_subset": True,
      "selection_seed": config.get("selection_seed"),
      "precision": config["precision"], "resolved_precision": config["precision"],
      "no_mixed_precision": False,
      **{key: config[key] for key in _MODEL_KEYS},
      "combo_alpha": 1.0 if arm == "plain_ce" else 0.4,
      "label_smoothing": 0.0 if arm == "plain_ce" else 0.05,
      "no_class_weights": arm == "plain_ce",
      "model_variant": arm if comparison == "architecture" else "cnn_ssm",
  }
  normalized = {**observed, "model_variant": observed.get("model_variant", "cnn_ssm"),
                "selection_seed": observed.get("selection_seed")}
  if any(normalized.get(key) != value for key, value in expected.items()):
    raise ValueError(f"Training configuration differs from plan: {directory}")
  manifest = training["manifest"]
  return manifest["epochs"][0]["paths"], manifest["validation_paths"]


def _verify_training(plan):
  config = plan["config"]
  arms = COMPARISON_ARMS[config.get("comparison", "objective")]
  for seed in config["seeds"]:
    left = _verify_run(config, seed, arms[0])
    right = _verify_run(config, seed, arms[1])
    if left != right:
      raise ValueError(f"Paired split memberships differ for seed {seed}")


def _provenance(root):
  source = Path(__file__).resolve().parents[2]
  snapshots = {
      "source_commit.txt": ["git", "-C", str(source), "rev-parse", "HEAD"],
      "source_worktree_status.txt": ["git", "-C", str(source), "status", "--short"],
      "packages.txt": [sys.executable, "-m", "pip", "freeze"],
  }
  for name, argv in snapshots.items():
    completed = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, check=False)
    (root / name).write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
      raise RuntimeError(f"Could not capture {name}; see {root / name}")


def _timestamp():
  return datetime.datetime.now(datetime.timezone.utc).isoformat()


def execute_plan(plan):
  """Runs once, preserving stage logs and atomic failure/completion records."""
  prepare_plan(plan)
  root = Path(plan["config"]["run_root"])
  sentinel = root / ".execution-started"
  with sentinel.open("x", encoding="utf-8") as handle:
    handle.write(_timestamp() + "\n")
  state = {
      "schema_version": 1, "state": "running", "started_at": _timestamp(),
      "stages": [{"id": step["id"], "state": "pending"} for step in plan["steps"]],
  }
  status = root / "execution.json"
  _atomic_json(status, state)
  try:
    _provenance(root)
    verified = False
    for step, stage in zip(plan["steps"], state["stages"]):
      if step["phase"] == "evaluate" and not verified:
        state["current_stage"] = "verify_training_artifacts"
        _atomic_json(status, state)
        _verify_training(plan)
        verified = True
      print(f"running {step['id']}", flush=True)
      state["current_stage"] = step["id"]
      stage.update({"state": "running", "started_at": _timestamp()})
      _atomic_json(status, state)
      log = Path(step["log_path"])
      log.parent.mkdir(parents=True, exist_ok=True)
      started = time.perf_counter()
      try:
        with log.open("x", encoding="utf-8") as handle:
          completed = subprocess.run(step["argv"], stdout=handle,
                                     stderr=subprocess.STDOUT, text=True, check=False)
        elapsed = time.perf_counter() - started
        stage.update({"seconds": elapsed, "returncode": completed.returncode})
        if "wall_seconds_path" in step:
          wall = Path(step["wall_seconds_path"])
          wall.parent.mkdir(parents=True, exist_ok=True)
          wall.write_text(f"{elapsed:.9f}\n", encoding="utf-8")
        if completed.returncode:
          raise RuntimeError(f"{step['id']} exited {completed.returncode}; see {log}")
        if step["phase"] == "train":
          _verify_run(plan["config"], step["seed"], step["arm"])
        if "artifact" in step:
          artifact = _read_object(step["artifact"])
          if step["phase"] == "preflight" and artifact.get("status") != "passed":
            raise ValueError(f"Preflight failed: {step['artifact']}")
        stage.update({"state": "completed", "finished_at": _timestamp()})
      except BaseException as error:
        stage.update({"state": "failed", "error": str(error),
                      "seconds": time.perf_counter() - started})
        raise
      finally:
        _atomic_json(status, state)
    state["state"] = "completed"
  except BaseException as error:
    failure = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
    state.update({"state": failure,
                  "error": str(error)})
    raise
  finally:
    state["finished_at"] = _timestamp()
    _atomic_json(status, state)
  return state


def main(argv=None):
  args = parse_args(argv)
  try:
    plan = make_plan(args)
    path = prepare_plan(plan)
    print(f"plan: {path}", flush=True)
    if args.execute:
      execute_plan(plan)
    else:
      print("No commands executed. Repeat the same arguments with --execute.",
            flush=True)
  except (OSError, ValueError, RuntimeError) as error:
    raise SystemExit(str(error)) from error


if __name__ == "__main__":
  main()
