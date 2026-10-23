"""Run an offline CPU demonstration on explicitly synthetic ForTy-format tiles.

Generates the actual 128x128 record schema, trains a small model for two epochs,
evaluates its checkpoint, and writes tile predictions and a comparison figure.
The synthetic scores demonstrate software integration, not remote-sensing
accuracy or improvements. No dataset downloads or GPU access are needed.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np

from mamba_forest.checkpoint import weights_sha256
from mamba_forest.constants import CLASS_NAMES, FEATURE_SHAPES, NUM_CLASSES

SCOPE = ("Synthetic CPU pipeline demonstration only. All inputs and labels are "
         "generated arrays, not satellite observations. Scores do not establish "
         "real-data accuracy, geographic generalization, or a model improvement.")
DEMO_MODEL = {"d_model": 4, "dropout": 0.0, "d_state_temporal": 2,
              "d_state_spatial": 2, "bottleneck_channels": 8}


def parse_args(argv=None) -> argparse.Namespace:
  """Parses local output settings and small synthetic sample/epoch budgets."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output-dir", type=Path,
                      help="New or empty directory; default: unique "
                           "runs/synthetic-demo-*")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--train-examples", type=int, default=4)
  parser.add_argument("--val-examples", type=int, default=2)
  parser.add_argument("--test-examples", type=int, default=2)
  parser.add_argument("--epochs", type=int, default=2)
  parser.add_argument("--batch-size", type=int, default=2)
  args = parser.parse_args(argv)
  budgets = ("train_examples", "val_examples", "test_examples", "epochs", "batch_size")
  for name in budgets:
    if getattr(args, name) < 1:
      parser.error(f"--{name.replace('_', '-')} must be positive")
  if not 0 <= args.seed <= 2**32 - 1:
    parser.error("--seed must be in [0, 2**32 - 1]")
  return args


def prepare_output_directory(path: Path | None) -> Path:
  """Claims a fresh output directory without overwriting any existing content."""
  if path is None:
    parent = Path("runs").resolve()
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="synthetic-demo-", dir=parent))
  path = path.expanduser().resolve()
  if path.exists():
    if not path.is_dir() or any(path.iterdir()):
      raise FileExistsError(f"{path} is not empty; choose a new --output-dir")
  else:
    path.mkdir(parents=True)
  return path


def _write_json(path, value):
  temporary = path.with_name(path.name + ".tmp")
  temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                       encoding="utf-8")
  temporary.replace(path)


def _sample(rng, prototypes):
  height, width = FEATURE_SHAPES["segmentation_labels"]
  yy, xx = np.indices((height, width))
  labels = ((yy * 3 // height) * 3 + xx * 3 // width).astype(np.int64)
  labels = np.roll(labels, tuple(rng.integers(0, height, size=2)), axis=(0, 1))
  seasonal = np.sin(np.arange(4) * np.pi / 2)[:, None, None, None] * 450
  optical = prototypes[labels][None, ...] + seasonal
  optical = np.clip(optical + rng.normal(0, 100, optical.shape), 0, 10000)
  elevation = np.stack([
      300 + labels * 260 + rng.normal(0, 30, labels.shape),
      np.clip(xx / width * 40 + rng.normal(0, 1, labels.shape), 0, 90),
      (yy / height * 360 + rng.uniform(0, 360)) % 360,
  ], axis=-1)
  climate = (rng.uniform(5, 100, (1, 14))
             + np.sin(np.arange(4)[:, None] * np.pi / 2)
             * rng.uniform(1, 10, (1, 14)) + rng.normal(0, 0.1, (4, 14)))
  return {"s2": optical.astype(np.float32),
          "elevation": elevation.astype(np.float32),
          "climate": climate.astype(np.float32), "segmentation_labels": labels}


def generate_synthetic_data(root: Path, seed: int, counts: dict) -> dict:
  """Writes distinct split samples in the real ForTy schema and raw test NPZs.

  Masks are constructed grids; reflectance-like arrays use synthetic class
  prototypes with seasonal offsets and noise. No physical realism is claimed.
  The test NPZs also retain their constructed labels for independent inspection.
  """
  import tensorflow as tf

  data_root, tiles_root = root / "data", root / "tiles"
  data_root.mkdir()
  tiles_root.mkdir()
  streams = np.random.SeedSequence(seed).spawn(4)
  prototypes = np.random.default_rng(streams[0]).uniform(500, 8500, (NUM_CLASSES, 10))
  manifest = {"schema_version": 1, "data_kind": "synthetic", "scope": SCOPE,
              "seed": seed, "feature_shapes": FEATURE_SHAPES, "splits": {},
              "test_tiles": []}
  for split, stream in zip(("train", "validation", "test"), streams[1:]):
    rng = np.random.default_rng(stream)
    path = data_root / f"forty_v1-{split}.tfrecord-00000-of-00001"
    samples = []
    with tf.io.TFRecordWriter(str(path)) as writer:
      for index in range(counts[split]):
        sample_id = f"synthetic_{split}_{index:03d}"
        example = _sample(rng, prototypes)
        features = {}
        for name, value in example.items():
          if name == "segmentation_labels":
            feature = tf.train.Feature(
                int64_list=tf.train.Int64List(value=value.reshape(-1)))
          else:
            feature = tf.train.Feature(
                float_list=tf.train.FloatList(value=value.reshape(-1)))
          features[name] = feature
        features["sample_id"] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[sample_id.encode("utf-8")]))
        serialized = tf.train.Example(
            features=tf.train.Features(feature=features)).SerializeToString()
        writer.write(serialized)
        samples.append({"sample_id": sample_id,
                        "record_sha256": hashlib.sha256(serialized).hexdigest()})
        if split == "test":
          tile_path = tiles_root / (sample_id + ".npz")
          np.savez_compressed(tile_path, **example)
          manifest["test_tiles"].append(str(tile_path))
    manifest["splits"][split] = {
        "path": str(path), "examples": counts[split], "samples": samples,
        "file_sha256": weights_sha256(path),
    }
  _write_json(data_root / "manifest.json", manifest)
  return manifest


def _source_info():
  repository = Path(__file__).resolve().parents[2]
  result = {"repository": str(repository), "python_version": platform.python_version()}
  commands = (("git_commit", ["rev-parse", "HEAD"]),
              ("git_status", ["status", "--short"]))
  for key, arguments in commands:
    try:
      completed = subprocess.run(["git", "-C", str(repository), *arguments],
                                 capture_output=True, text=True, check=False)
      result[key] = completed.stdout.strip() if completed.returncode == 0 else None
      if completed.returncode:
        result[key + "_error"] = completed.stderr.strip()
    except OSError as error:
      result[key + "_error"] = str(error)
  result["source_files_sha256"] = {
      path.name: weights_sha256(path) for path in Path(__file__).parent.glob("*.py")}
  result["package_versions"] = {}
  for name in ("tensorflow", "keras", "numpy", "matplotlib"):
    try:
      result["package_versions"][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
      result["package_versions"][name] = None
  return result


def _commands(args, root):
  source = ["--data-root", str(root / "data"), "--total-shards", "1"]
  batch = ["--batch-size", str(args.batch_size)]
  weights = root / "training" / "best.weights.h5"
  model = [value for name, setting in DEMO_MODEL.items()
           for value in ("--" + name.replace("_", "-"), str(setting))]
  return {
      "train": [sys.executable, "-m", "mamba_forest.train", *source, *batch, *model,
                "--train-shards", "1", "--val-shards", "1", "--fixed-train-subset",
                "--epochs", str(args.epochs), "--seed", str(args.seed),
                "--patience", "0", "--precision", "float32",
                "--output-dir", str(root / "training")],
      "evaluate": [sys.executable, "-m", "mamba_forest.evaluate", *source, *batch,
                   "--weights", str(weights), "--test-shards", "1",
                   "--output-dir", str(root / "evaluation")],
      "predict": [sys.executable, "-m", "mamba_forest.predict",
                  "--weights", str(weights),
                  "--input-dir", str(root / "tiles"), "--output-dir",
                  str(root / "predictions"), "--save-png", "--timings",
                  str(root / "predictions" / "timings.json")],
  }


def _run_stage(command, log_path, environment):
  with log_path.open("x", encoding="utf-8") as handle:
    completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                               text=True, env=environment, check=False)
  if completed.returncode:
    raise RuntimeError(f"stage exited {completed.returncode}; see {log_path}")


def _collect_results(root, args, manifest):
  training = root / "training"
  checkpoint = training / "best.weights.h5"
  digest = weights_sha256(checkpoint)
  selection = json.loads((training / "checkpoint_selection.json").read_text())
  metrics = json.loads((root / "evaluation" / "test_metrics.json").read_text())
  config = json.loads((training / "config.json").read_text())
  if selection["weights_sha256"] != digest or metrics["weights_sha256"] != digest:
    raise ValueError("selected, evaluated, and actual checkpoint hashes differ")
  if config["resolved_precision"] != "float32":
    raise ValueError("the synthetic CPU demo requires float32 training")
  if any(device["type"] == "GPU" for device in metrics["runtime"]["devices"]):
    raise ValueError("the synthetic CPU demo unexpectedly exposed a GPU")
  with (training / "training_results.csv").open(newline="") as handle:
    history = list(csv.DictReader(handle))
  if [int(row["epoch"]) for row in history] != list(range(1, args.epochs + 1)):
    raise ValueError("training did not complete the requested epochs")
  expected_pixels = args.test_examples * np.prod(FEATURE_SHAPES["segmentation_labels"])
  if metrics["num_pixels"] != expected_pixels:
    raise ValueError("synthetic evaluation pixel count differs from generated test set")
  with (root / "predictions" / "predictions.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))
  expected_stems = [Path(path).stem for path in manifest["test_tiles"]]
  if [row["tile"] for row in rows] != expected_stems:
    raise ValueError("prediction rows differ from the generated test tiles")
  for stem in expected_stems:
    predicted = np.load(root / "predictions" / f"{stem}_class_map.npy",
                        allow_pickle=False)
    if (predicted.shape != FEATURE_SHAPES["segmentation_labels"]
            or predicted.dtype != np.uint8 or predicted.max() >= NUM_CLASSES):
      raise ValueError(f"invalid synthetic prediction map for {stem}")
    if not (root / "predictions" / f"{stem}_class_map.png").is_file():
      raise ValueError(f"missing prediction PNG for {stem}")
  timings_path = root / "predictions" / "timings.json"
  json.loads(timings_path.read_text())
  return {"checkpoint": {"path": str(checkpoint), "sha256": digest,
                         "selection_path": str(training / "checkpoint_selection.json")},
          "completed_epochs": len(history), "evaluation_runtime": metrics["runtime"],
          "synthetic_scores": {name: metrics[name] for name in (
              "pixel_accuracy", "mean_iou", "overall_f1", "forests_f1")},
          "prediction_tiles": len(rows), "prediction_timings_path": str(timings_path)}


def render_comparison(root, manifest):
  """Draws generated optical illustrations, constructed labels, and predictions."""
  import matplotlib
  matplotlib.use("Agg")
  from matplotlib import pyplot as plt
  from matplotlib.patches import Patch

  from mamba_forest.predict import CLASS_COLORS, colourise

  figure, axes = plt.subplots(len(manifest["test_tiles"]), 3,
                              figsize=(11, 3 * len(manifest["test_tiles"]) + 2),
                              squeeze=False)
  try:
    for row, path in enumerate(manifest["test_tiles"]):
      path = Path(path)
      with np.load(path, allow_pickle=False) as tile:
        optical = np.clip(tile["s2"][0, ..., :3] / 10000, 0, 1)
        labels = tile["segmentation_labels"]
      predicted = np.load(root / "predictions" / f"{path.stem}_class_map.npy",
                          allow_pickle=False)
      images = (optical, colourise(labels), colourise(predicted))
      titles = (
          f"{path.stem}\nSynthetic optical illustration", "Constructed class labels",
          "Model prediction on synthetic data")
      for axis, image, title in zip(axes[row], images, titles):
        axis.imshow(image)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    figure.suptitle("Synthetic forest-type segmentation pipeline\n"
                    "Generated arrays only; no satellite imagery "
                    "or real-data accuracy claim",
                    fontsize=12)
    legend = [Patch(facecolor=np.asarray(color) / 255, label=name)
              for color, name in zip(CLASS_COLORS, CLASS_NAMES)]
    figure.legend(handles=legend, loc="lower center", ncol=3, fontsize=8)
    figure.subplots_adjust(bottom=0.16, top=0.86, hspace=0.4, wspace=0.08)
    figure.savefig(root / "comparison.png", dpi=140, bbox_inches="tight")
  finally:
    plt.close(figure)


def run_demo(args: argparse.Namespace) -> dict:
  """Runs once and preserves stage logs and summary.json on ordinary failures."""
  root = prepare_output_directory(args.output_dir)
  summary_path = root / "summary.json"
  summary = {"schema_version": 1, "status": "running", "scope": SCOPE,
             "data_kind": "synthetic", "output_dir": str(root), "seed": args.seed,
             "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "model": DEMO_MODEL, "device": "CPU", "precision": "float32",
             "tile_shape": FEATURE_SHAPES["segmentation_labels"],
             "requested_epochs": args.epochs, "batch_size": args.batch_size,
             "counts": {"train": args.train_examples, "validation": args.val_examples,
                        "test": args.test_examples}, "stages": [],
             "outputs": {"data_manifest": str(root / "data" / "manifest.json"),
                         "training": str(root / "training"),
                         "evaluation": str(root / "evaluation" / "test_metrics.json"),
                         "predictions": str(root / "predictions"),
                         "comparison": str(root / "comparison.png")}}
  _write_json(summary_path, summary)
  print(f"Synthetic CPU demo: {root}\n{SCOPE}", flush=True)
  cache_defaults = {"MPLCONFIGDIR": root / "runtime" / "matplotlib",
                    "XDG_CACHE_HOME": root / "runtime" / "cache"}
  previous_caches = {key: os.environ.get(key) for key in cache_defaults}
  try:
    # Keras can import Matplotlib during the first TensorFlow import. Keep its
    # caches writable before generating records, then restore the caller's env.
    for key, path in cache_defaults.items():
      if previous_caches[key] is None:
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    summary["current_stage"] = "generate_data"
    summary["source"] = _source_info()
    manifest = generate_synthetic_data(root, args.seed, summary["counts"])
    environment = dict(os.environ)
    environment.update({"CUDA_VISIBLE_DEVICES": "-1", "TF_NUM_INTRAOP_THREADS": "2",
                        "TF_NUM_INTEROP_THREADS": "2", "TF_CPP_MIN_LOG_LEVEL": "2"})
    recorded_environment = ("CUDA_VISIBLE_DEVICES", "TF_NUM_INTRAOP_THREADS",
                            "TF_NUM_INTEROP_THREADS", "MPLCONFIGDIR", "XDG_CACHE_HOME")
    summary["command_environment"] = {
        key: environment[key] for key in recorded_environment}
    logs = root / "logs"
    logs.mkdir()
    for name, command in _commands(args, root).items():
      stage = {"name": name, "state": "running", "argv": command,
               "log_path": str(logs / (name + ".log"))}
      summary["current_stage"] = name
      summary["stages"].append(stage)
      _write_json(summary_path, summary)
      print(f"running synthetic {name}; log: {stage['log_path']}", flush=True)
      started = time.perf_counter()
      _run_stage(command, Path(stage["log_path"]), environment)
      stage.update(state="completed", seconds=time.perf_counter() - started)
    summary["current_stage"] = "verify_outputs"
    summary.update(_collect_results(root, args, manifest))
    summary["current_stage"] = "render_comparison"
    render_comparison(root, manifest)
    summary["status"] = "completed"
  except BaseException as error:
    summary["status"] = ("interrupted" if isinstance(error, KeyboardInterrupt)
                         else "failed")
    summary["error"] = {"type": type(error).__name__, "message": str(error),
                        "stage": summary.get("current_stage")}
    if summary["stages"] and summary["stages"][-1]["state"] == "running":
      summary["stages"][-1]["state"] = "failed"
    raise
  finally:
    for key, previous in previous_caches.items():
      if previous is None:
        os.environ.pop(key, None)
      else:
        os.environ[key] = previous
    summary["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    _write_json(summary_path, summary)
  print(f"Synthetic demo completed: {summary_path}", flush=True)
  return summary


def main() -> None:
  args = parse_args()
  try:
    run_demo(args)
  except (OSError, ValueError, RuntimeError) as error:
    raise SystemExit(str(error)) from error


if __name__ == "__main__":
  main()
