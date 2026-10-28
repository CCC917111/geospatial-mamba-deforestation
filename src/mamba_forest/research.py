"""Retain and continue fixed-cohort, multi-seed architecture pilot stages.

Continuation skips only verified completed stages. An interrupted training
stage is preserved as an attempt and restarted from epoch one, without loading
optimizer state. All training must validate before any test evaluation.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid

from mamba_forest import compare, experiment, stage_data
from mamba_forest.constants import GCS_ROOT, TOTAL_SHARDS


RUNTIME_PROBE = """import importlib.metadata as m, json, platform, sys
import numpy as np
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if sys.argv[1] == 'gpu' and not gpus:
    raise RuntimeError('A GPU is required for this real-data pilot')
tf.config.set_soft_device_placement(False)
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    value = tf.matmul(tf.ones((2, 2)), tf.ones((2, 2)))
value.numpy()
expected = 'GPU:0' if gpus else 'CPU:0'
if not value.device.endswith(expected):
    raise RuntimeError('Runtime probe silently changed device')
build = tf.sysconfig.get_build_info()
print(json.dumps({'tensorflow':tf.__version__, 'keras':m.version('keras'),
    'numpy':np.__version__, 'python':platform.python_version(),
    'probe_device':value.device, 'gpu_details':[
        tf.config.experimental.get_device_details(gpu) for gpu in gpus],
    'build':{key:build.get(key) for key in ('cuda_version','cudnn_version')}}))
"""


def digest(path):
  result = hashlib.sha256()
  with Path(path).open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      result.update(block)
  return result.hexdigest()


def read(path):
  value = json.loads(Path(path).read_text())
  if not isinstance(value, dict):
    raise ValueError(f"Expected a JSON object: {path}")
  return value


def save(path, value):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  experiment._atomic_json(path, value)


def identifier():
  stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  return stamp + "-" + uuid.uuid4().hex[:8]


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run-root", type=Path, required=True)
  parser.add_argument("--data-root", required=True)
  parser.add_argument("--data-manifest", type=Path, required=True)
  parser.add_argument("--data-kind", choices=("real_forty", "synthetic"),
                      default="real_forty")
  parser.add_argument("--execute", action="store_true")
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--recover-lock", action="store_true")
  parser.add_argument("--allow-cpu", action="store_true")
  parser.add_argument("--max-train-runs", type=int)
  parser.add_argument("--stop-after-training", action="store_true")
  parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
  parser.add_argument("--selection-seed", type=int, default=2026)
  for name, default in (("total-shards", TOTAL_SHARDS), ("train-shards", 8),
                        ("val-shards", 4), ("test-shards", 8), ("epochs", 10),
                        ("batch-size", 8)):
    parser.add_argument("--" + name, type=int, default=default)
  parser.add_argument("--precision", choices=("auto", "float32", "mixed_float16"),
                      default="auto")
  for name, key in experiment._MODEL_KEYS.items():
    parser.add_argument("--" + name.replace("_", "-"),
                        type=float if name == "dropout" else int,
                        default=experiment.MODEL_DEFAULTS[key])
  args = parser.parse_args(argv)
  if args.max_train_runs is not None and args.max_train_runs < 0:
    parser.error("--max-train-runs must be nonnegative")
  if args.data_kind == "real_forty":
    if args.allow_cpu or len(args.seeds) < 3 or args.total_shards != TOTAL_SHARDS:
      parser.error("real ForTy requires GPU, at least three seeds and 1024 namespaces")
  return args


def make_plan(args):
  arguments = ["--run-root", str(args.run_root), "--comparison", "architecture",
               "--data-root", args.data_root, "--precision", args.precision,
               "--seeds", *map(str, args.seeds)]
  for name in ("total_shards", "train_shards", "val_shards", "test_shards",
               "epochs", "batch_size", "selection_seed", *experiment._MODEL_KEYS):
    arguments += ["--" + name.replace("_", "-"), str(getattr(args, name))]
  if args.allow_cpu:
    arguments.append("--allow-cpu")
  plan = experiment.make_plan(experiment.parse_args(arguments))
  train = {step["id"]: step for step in plan["steps"] if step["phase"] == "train"}
  arms = compare.COMPARISON_ARMS["architecture"]
  ordered = []
  for index, seed in enumerate(args.seeds):
    for arm in arms if index % 2 == 0 else arms[::-1]:
      ordered.append(train[f"train_{arm}_{seed}"])
  plan["steps"] = ([step for step in plan["steps"] if step["phase"] == "preflight"]
                   + ordered + [step for step in plan["steps"]
                                if step["phase"] in ("evaluate", "compare")])
  plan["research_controls"] = {
      "shared_selection_seed": args.selection_seed,
      "arm_order": "alternates across seeds",
      "continuation": "verified completed stages only; partial training restarts",
      "class_semantics": "unverified; primary report uses numeric IDs 0 through 8"}
  return plan


def _prepare_locked(plan, args):
  root = Path(plan["config"]["run_root"])
  root.mkdir(parents=True, exist_ok=True)
  spec = {"schema_version": 1, "protocol": "fixed-cohort-architecture-pilot-v1",
          "data_kind": args.data_kind, "config": plan["config"],
          "data_manifest": str(args.data_manifest.expanduser().resolve()),
          "class_semantics": plan["research_controls"]["class_semantics"],
          "checkpoint_policy": "validation macro-F1 over all nine numeric IDs"}
  path = root / "research_spec.json"
  manifest = args.data_manifest.expanduser().resolve()
  reserved = {"plan.json", "research_spec.json", "research_state.json",
              "source_identity.json", "runtime_identity.json", "data_identity.json",
              "checkpoint_inventory.json", "checkpoint_manifest.json",
              "research_report.json", "comparison.json", ".research.lock",
              ".research.recovery"}
  if manifest.parent == root and manifest.name in reserved:
    raise ValueError("Data manifest conflicts with a reserved research output")
  if path.exists():
    if read(path) != spec or read(root / "plan.json") != plan:
      raise ValueError("Study protocol changed; use a new run root")
  else:
    def allowed(item):
      return (item.name == ".research.lock" or item == manifest
              and item.is_file() and not item.is_symlink())
    if any(not allowed(item) for item in root.iterdir()):
      raise ValueError("Existing artifacts lack a research protocol; use a fresh root")
    save(root / "plan.json", plan)
    save(path, spec)
  if (root / "research_state.json").exists() and not args.resume:
    raise ValueError("Study has execution state; use --resume to verify and continue")
  return root


def prepare(plan, args):
  root = Path(plan["config"]["run_root"])
  root.mkdir(parents=True, exist_ok=True)
  with study_lock(root, args.recover_lock):
    return _prepare_locked(plan, args)


@contextmanager
def recovery_guard(root):
  path = root / ".research.recovery"
  try:
    with path.open("x") as handle:
      handle.write("Exclusive lock-recovery reservation\n")
  except FileExistsError as error:
    raise ValueError(
        "Lock recovery is active or was interrupted; stop the prior runtime "
        "before removing only .research.recovery") from error
  try:
    yield
  finally:
    path.unlink()


@contextmanager
def study_lock(root, recover=False):
  path = root / ".research.lock"
  token = identifier()
  if path.exists():
    observed = path.read_bytes()
    try:
      previous = json.loads(observed)
      if not isinstance(previous, dict):
        previous = {}
    except (ValueError, OSError):
      previous = {}
    pid = previous.get("pid")
    same_host = (previous.get("host") == platform.node() and isinstance(pid, int)
                 and not isinstance(pid, bool) and pid > 0)
    if same_host:
      try:
        os.kill(pid, 0)
      except ProcessLookupError:
        pass
      else:
        raise ValueError("A study process is still running on this host")
    if not recover:
      raise ValueError(
          "Study lock remains; stop the prior runtime, then use --recover-lock")
    with recovery_guard(root):
      if not path.exists() or path.read_bytes() != observed:
        raise ValueError("Study lock changed during recovery; no takeover performed")
      archive = root / "attempts" / ("recovered-lock-" + token + ".txt")
      archive.parent.mkdir(exist_ok=True)
      path.replace(archive)
      with path.open("x") as handle:
        json.dump({"token": token, "host": platform.node(), "pid": os.getpid()}, handle)
  else:
    with path.open("x") as handle:
      json.dump({"token": token, "host": platform.node(), "pid": os.getpid()}, handle)
  try:
    yield
  finally:
    if path.exists() and read(path).get("token") == token:
      path.unlink()


def source_snapshot():
  directory = Path(__file__).resolve().parent
  root = directory.parents[1]

  def git(*arguments):
    return subprocess.check_output(["git", "-C", str(root), *arguments],
                                   text=True).strip()
  files = ("research.py", "research_report.py", "stage_data.py", "experiment.py",
           "compare.py", "data.py", "train.py", "evaluate.py", "checkpoint.py",
           "model.py", "layers.py", "losses.py", "constants.py")
  return {"commit": git("rev-parse", "HEAD"),
          "worktree_status": git("status", "--porcelain"),
          "module_sha256": {name: digest(directory / name) for name in files}}


def checked_command(command, log_path):
  """Fresh worker inherits the notebook driver's process group."""
  log_path.parent.mkdir(parents=True, exist_ok=True)
  with log_path.open("x") as log:
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                               text=True)
    try:
      code = process.wait()
    finally:
      if process.poll() is None:
        process.terminate()
        try:
          process.wait(timeout=10)
        except subprocess.TimeoutExpired:
          process.kill()
          process.wait()
  if code:
    raise RuntimeError(f"Worker exited {code}; inspect {log_path}")


def freeze(path, value, label):
  if path.exists():
    if read(path) != value:
      raise ValueError(f"{label} differs from the frozen study identity")
  else:
    save(path, value)


def validate_data(path, config, data_kind):
  manifest = read(path)
  expected = {key: config[key] for key in (
      "total_shards", "train_shards", "val_shards", "test_shards", "selection_seed")}
  expected["cache_root"] = config["data_root"]
  observed = manifest.get("config", {})
  invalid = (manifest.get("schema_version") != 1 or manifest.get("status") != "ready"
             or any(observed.get(key) != value for key, value in expected.items()))
  if invalid:
    raise ValueError("Staged data differs from the research protocol")
  if data_kind == "real_forty" and observed.get("source_root") != GCS_ROOT:
    raise ValueError("Real-data scope requires the recorded official ForTy source")
  selection = {key: expected[key] for key in expected if key != "cache_root"}
  expected_files = stage_data.selected_files(observed["source_root"], **selection)
  files = manifest.get("files", [])
  if len(files) != len(expected_files):
    raise ValueError("Staged data manifest is incomplete")
  for record, selected in zip(files, expected_files):
    if any(record.get(key) != value for key, value in selected.items()):
      raise ValueError("Staged data membership or source differs")
    local = Path(config["data_root"]) / selected["filename"]
    changed = (local.is_symlink() or not local.is_file()
               or local.stat().st_size != record["bytes"]
               or digest(local) != record["sha256"])
    if changed:
      raise ValueError("Staged data bytes changed; rerun the staging check")
  return {"manifest_sha256": digest(path), "config": observed,
          "files": files, "data_kind": data_kind}


def start_session(root, config, args):
  session = root / "sessions" / identifier()
  session.mkdir(parents=True)
  source = source_snapshot()
  save(session / "source.json", source)
  if args.data_kind == "real_forty" and source["worktree_status"]:
    raise ValueError("Real-data study requires an unmodified pinned source checkout")
  freeze(root / "source_identity.json", {
      key: source[key] for key in ("commit", "module_sha256")}, "Source")
  checked_command([sys.executable, "-c", RUNTIME_PROBE,
                   "cpu" if config["allow_cpu"] else "gpu"], session / "runtime.log")
  runtime = json.loads((session / "runtime.log").read_text().splitlines()[-1])
  save(session / "runtime.json", runtime)
  freeze(root / "runtime_identity.json", runtime, "Runtime")
  data = validate_data(args.data_manifest, config, args.data_kind)
  freeze(root / "data_identity.json", data, "Dataset")
  save(session / "data_stats.json", data_stats(data))
  packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
  (session / "packages.txt").write_text(packages)
  return session.name


def data_stats(identity):
  files = {}
  for record in identity["files"]:
    path = Path(identity["config"]["cache_root"]) / record["filename"]
    value = path.stat()
    files[record["filename"]] = {"bytes": value.st_size,
                                 "mtime_ns": value.st_mtime_ns, "inode": value.st_ino}
  return files


def verify_session_identity(root, data_kind):
  source = source_snapshot()
  current = {key: source[key] for key in ("commit", "module_sha256")}
  if current != read(root / "source_identity.json"):
    raise ValueError("Source changed during this session")
  if data_kind == "real_forty" and source["worktree_status"]:
    raise ValueError("Real-data source was modified during this session")
  runtime = read(root / "runtime_identity.json")
  versions = {"keras": importlib.metadata.version("keras"),
              "numpy": importlib.metadata.version("numpy"),
              "python": platform.python_version()}
  for name in ("tensorflow", "tensorflow-cpu", "tf-nightly", "tf-nightly-cpu"):
    try:
      versions["tensorflow"] = importlib.metadata.version(name)
      break
    except importlib.metadata.PackageNotFoundError:
      continue
  if "tensorflow" not in versions:
    raise ValueError("TensorFlow distribution metadata is unavailable")
  if any(runtime.get(key) != value for key, value in versions.items()):
    raise ValueError("Software versions changed during this session")
  data = read(root / "data_identity.json")
  spec = read(root / "research_spec.json")
  if digest(spec["data_manifest"]) != data["manifest_sha256"]:
    raise ValueError("Dataset manifest changed during this session")
  session = read(root / "research_state.json")["session"]
  if data_stats(data) != read(root / "sessions" / session / "data_stats.json"):
    raise ValueError("Staged data files changed during this session")


def artifact_record(root, path):
  if path.is_symlink() or not path.is_file() or not path.stat().st_size:
    raise ValueError(f"Missing, empty or unsafe artifact: {path}")
  return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
          "sha256": digest(path)}


def stage_outputs(root, step, plan):
  if step["phase"] == "train":
    experiment._verify_run(plan["config"], step["seed"], step["arm"])
    directory = root / f"{step['arm']}_{step['seed']}"
    observed = read(directory / "config.json")["tensorflow_version"]
    if observed != read(root / "runtime_identity.json")["tensorflow"]:
      raise ValueError("Training runtime differs from the frozen study runtime")
    names = ("config.json", "best.weights.h5", "last.weights.h5",
             "training_results.csv", "split_manifest.json", "checkpoint_selection.json",
             "train_wall_seconds.txt")
    return [artifact_record(root, directory / name) for name in names]
  if step["phase"] == "evaluate":
    # The existing strict comparator verifies the confusion matrix and checkpoint.
    arm, seed = step["id"].removeprefix("evaluate_").rsplit("_", 1)
    compare._read_run(root / f"{arm}_{seed}", arm, int(seed))
  artifact = Path(step["artifact"])
  value = read(artifact)
  if step["phase"] == "preflight" and value.get("status") != "passed":
    raise ValueError("Data preflight did not pass")
  outputs = [artifact_record(root, artifact)]
  if "wall_seconds_path" in step:
    outputs.append(artifact_record(root, Path(step["wall_seconds_path"])))
  return outputs


def verify_receipt(root, step, stage, plan):
  receipt_path = root / "receipts" / (step["id"] + ".json")
  receipt = read(receipt_path)
  if receipt.get("stage_id") != step["id"] or receipt.get("argv") != step["argv"]:
    raise ValueError("Completed stage receipt differs from the plan")
  if receipt.get("outputs") != stage_outputs(root, step, plan):
    raise ValueError("Completed stage artifacts changed; no automatic overwrite")
  if stage.get("receipt_sha256") != digest(receipt_path):
    raise ValueError("Completed stage receipt changed")


def preserve_partial(root, step):
  target = (root / f"{step['arm']}_{step['seed']}" if step["phase"] == "train"
            else Path(step["artifact"]).parent if step["phase"] == "evaluate"
            else Path(step["artifact"]))
  wall = Path(step["wall_seconds_path"]) if "wall_seconds_path" in step else None
  receipt = root / "receipts" / (step["id"] + ".json")
  if target.exists() or receipt.exists() or wall is not None and wall.exists():
    destination = root / "attempts" / (step["id"] + "-" + identifier())
    destination.mkdir(parents=True)
    if target.exists():
      target.replace(destination / target.name)
    if wall is not None and wall.exists():
      wall.replace(destination / wall.name)
    if receipt.exists():
      receipt.replace(destination / (step["id"] + "-receipt.json"))
    return str(destination.relative_to(root))
  return None


def verify_all_training(root, plan, state):
  reference = None
  for step in plan["steps"]:
    if step["phase"] != "train":
      continue
    stage = state["stages"][step["id"]]
    if stage["state"] != "completed":
      raise ValueError("All training must complete before any test evaluation")
    verify_receipt(root, step, stage, plan)
    membership = experiment._verify_run(plan["config"], step["seed"], step["arm"])
    if reference is None:
      reference = membership
    elif reference != membership:
      raise ValueError("Training cohorts differ across models or seeds")


def checkpoint_inventory(root, plan, state):
  runs = []
  for step in plan["steps"]:
    completed = state["stages"][step["id"]]["state"] == "completed"
    if step["phase"] == "train" and completed:
      directory = root / f"{step['arm']}_{step['seed']}"
      epoch = read(directory / "checkpoint_selection.json")["epoch"]
      runs.append({"seed": step["seed"], "variant": step["arm"],
                   "selected_epoch": epoch,
                   "best": artifact_record(root, directory / "best.weights.h5"),
                   "last": artifact_record(root, directory / "last.weights.h5")})
  save(root / "checkpoint_inventory.json", {
      "schema_version": 1, "completed_training_runs": len(runs),
      "required_training_runs": 2 * len(plan["config"]["seeds"]), "runs": runs,
      "scope": "Weights-only checkpoints; optimizer state is not resumed"})


def execute(plan, args):
  root = Path(plan["config"]["run_root"])
  root.mkdir(parents=True, exist_ok=True)
  status_path = root / "research_state.json"
  with study_lock(root, args.recover_lock):
    _prepare_locked(plan, args)
    if status_path.exists():
      identities = ("source_identity.json", "runtime_identity.json",
                    "data_identity.json")
      for name in identities:
        path = root / name
        if not path.is_file() or path.is_symlink():
          raise ValueError(
              "A frozen study identity is missing; no replacement inferred")
    else:
      output_dirs = [root / f"{arm}_{seed}" for seed in plan["config"]["seeds"]
                     for arm in compare.COMPARISON_ARMS["architecture"]]
      if (root / "receipts").exists() or any(path.exists() for path in output_dirs):
        raise ValueError("Execution ledger is missing despite existing study outputs")
    session = start_session(root, plan["config"], args)
    state = read(status_path) if status_path.exists() else {
        "schema_version": 1,
        "stages": {step["id"]: {"state": "pending", "attempts": []}
                   for step in plan["steps"]}}
    if set(state["stages"]) != {step["id"] for step in plan["steps"]}:
      raise ValueError("Execution stages differ from the frozen plan")
    if any(stage.get("state") not in ("pending", "running", "completed", "failed")
           for stage in state["stages"].values()):
      raise ValueError("Unknown saved stage state; no automatic restart")
    state.update({"state": "running", "session": session})
    state.pop("error", None)
    save(status_path, state)
    trained, verified = 0, False
    try:
      for step in plan["steps"]:
        stage = state["stages"][step["id"]]
        if step["phase"] == "evaluate" and not verified:
          verify_all_training(root, plan, state)
          identity = validate_data(args.data_manifest, plan["config"], args.data_kind)
          if identity != read(root / "data_identity.json"):
            raise ValueError("Dataset changed before test evaluation")
          verified = True
          if args.stop_after_training:
            state["state"] = "training_ready_for_evaluation"
            break
        if stage["state"] == "completed":
          verify_receipt(root, step, stage, plan)
          print("verified", step["id"], flush=True)
          continue
        limit = (step["phase"] == "train" and args.max_train_runs is not None
                 and trained >= args.max_train_runs)
        if limit:
          state["state"] = "training_pending"
          break
        preserved = preserve_partial(root, step)
        attempt = {"id": identifier(), "session": session,
                   "preserved_partial": preserved, "state": "running"}
        stage["attempts"].append(attempt)
        stage["state"] = "running"
        state["current_stage"] = step["id"]
        save(status_path, state)
        log = root / "logs" / (step["id"] + "-" + attempt["id"] + ".log")
        attempt["log"] = str(log.relative_to(root))
        print("running", step["id"], flush=True)
        started = time.perf_counter()
        try:
          verify_session_identity(root, args.data_kind)
          checked_command(step["argv"], log)
          verify_session_identity(root, args.data_kind)
          attempt["seconds"] = time.perf_counter() - started
          if "wall_seconds_path" in step:
            Path(step["wall_seconds_path"]).write_text(f"{attempt['seconds']:.9f}\n")
          outputs = stage_outputs(root, step, plan)
          attempt["state"] = "completed"
          receipt = {"schema_version": 1, "stage_id": step["id"],
                     "argv": step["argv"], "session": session,
                     "outputs": outputs, "attempt": attempt}
          receipt_path = root / "receipts" / (step["id"] + ".json")
          save(receipt_path, receipt)
          stage.update({"state": "completed", "receipt_sha256": digest(receipt_path)})
          if step["phase"] == "train":
            trained += 1
          checkpoint_inventory(root, plan, state)
        except BaseException as error:
          stage["state"] = attempt["state"] = "failed"
          attempt["error"] = f"{type(error).__name__}: {error}"
          raise
        finally:
          save(status_path, state)
      else:
        verify_session_identity(root, args.data_kind)
        from mamba_forest.research_report import build_report
        build_report(root, plan["config"]["seeds"], data_manifest=args.data_manifest)
        state["state"] = "completed"
    except BaseException as error:
      state.update({"state": "interrupted" if isinstance(error, KeyboardInterrupt)
                    else "failed", "error": f"{type(error).__name__}: {error}"})
      raise
    finally:
      save(status_path, state)
    return state


def main(argv=None):
  args = parse_args(argv)
  plan = make_plan(args)
  if args.execute:
    state = execute(plan, args)
    print("Research state:", state["state"], flush=True)
  else:
    root = prepare(plan, args)
    print("Frozen research plan:", root / "plan.json", flush=True)


if __name__ == "__main__":
  main()
