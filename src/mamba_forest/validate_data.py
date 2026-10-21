"""Inspect one raw ForTy record per split before a real-data training run.

Example:
    python -m mamba_forest.validate_data --require-gpu --output preflight.json

Only record 0 of one selected shard in each split is inspected. Passing this
check establishes neither full-dataset validity nor geographic isolation.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import tensorflow as tf

from mamba_forest import data as data_lib

_ID_KEYS = {"id", "sample_id", "tile_id", "example_id", "patch_id"}
_COORDINATE_KEYS = {"lat", "lon", "latitude", "longitude"}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--data-root", "--gcs-root", dest="gcs_root",
                      default=data_lib.GCS_ROOT,
                      help="ForTy shard root; a local directory is also supported")
  parser.add_argument("--train-shards", type=int, default=128)
  parser.add_argument("--val-shards", type=int, default=8)
  parser.add_argument("--test-shards", type=int, default=64)
  parser.add_argument("--total-shards", type=int, default=data_lib.TOTAL_SHARDS)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--require-gpu", action="store_true",
                      help="Reject CPU-only runtimes before opening any shard")
  parser.add_argument("--output", default="preflight.json",
                      help="JSON report written on both success and failure")
  return parser.parse_args()


def _metadata_values(feature, kind):
  values = []
  for value in getattr(feature, kind).value:
    if isinstance(value, bytes):
      try:
        value = value.decode("utf-8")
      except UnicodeDecodeError:
        value = "hex:" + value.hex()
    elif isinstance(value, float) and not math.isfinite(value):
      value = str(value)
    values.append(value)
  return values


def inspect_record(path: str, result: dict) -> None:
  """Records diagnostics while validating record 0 with the existing pipeline."""
  records = tf.data.TFRecordDataset(path, buffer_size=64 * 1024).take(1)
  raw = next(iter(records), None)
  if raw is None:
    raise ValueError("shard contains no records")
  example = tf.train.Example.FromString(bytes(raw.numpy()))
  result.update({"raw_features": {}, "metadata": {}})
  for key, feature in sorted(example.features.feature.items()):
    kind = feature.WhichOneof("kind")
    result["raw_features"][key] = {
        "kind": kind,
        "length": len(getattr(feature, kind).value) if kind else 0,
    }
    if key.rsplit("/", 1)[-1].lower() in _ID_KEYS | _COORDINATE_KEYS and kind:
      result["metadata"][key] = _metadata_values(feature, kind)

  parsed = data_lib.parse_example(raw)
  normalized = data_lib.normalize(parsed)
  encoded = data_lib.encode_labels(normalized)
  result["tensors"] = {}
  for stage, tensors in (("parsed", parsed), ("normalized", normalized),
                         ("encoded", encoded)):
    result["tensors"][stage] = {
        key: {"shape": tensor.shape.as_list(), "dtype": tensor.dtype.name}
        for key, tensor in tensors.items()
    }
  result["finite_inputs"] = {
      key: bool(tf.reduce_all(tf.math.is_finite(encoded[key])))
      for key in ("s2", "elevation", "climate")
  }
  labels = encoded["segmentation_labels"].numpy()
  result["labels"] = {
      "min": int(labels.min()),
      "max": int(labels.max()),
      "num_pixels": int(labels.size),
      "class_support": dict(zip(
          data_lib.CLASS_NAMES,
          np.bincount(labels.reshape(-1), minlength=data_lib.NUM_CLASSES).tolist())),
  }
  if not all(result["finite_inputs"].values()):
    invalid = [key for key, finite in result["finite_inputs"].items() if not finite]
    raise ValueError(f"non-finite normalized inputs: {', '.join(invalid)}")


def run_preflight(args: argparse.Namespace) -> dict:
  """Returns a pass/fail report, stopping at the first failed split."""
  devices = tf.config.list_physical_devices()
  report = {
      "status": "failed",
      "scope": "First record of one selected shard per split only; this does "
               "not establish geographic isolation or full-dataset validity.",
      "config": vars(args).copy(),
      "runtime": {
          "tensorflow_version": tf.__version__,
          "precision_policy": tf.keras.mixed_precision.global_policy().name,
          "devices": [{"name": device.name, "type": device.device_type}
                      for device in devices],
      },
      "splits": {},
      "errors": [],
  }
  if args.require_gpu and not any(device.device_type == "GPU" for device in devices):
    report["errors"].append({
        "stage": "gpu", "error": "--require-gpu requested but no GPU is visible"})
    return report

  seen_ids = {}
  budgets = (("train", args.train_shards), ("validation", args.val_shards),
             ("test", args.test_shards))
  for split, budget in budgets:
    inspected = {"status": "failed", "record_index": 0}
    report["splits"][split] = inspected
    try:
      selected = data_lib.selected_shard_paths(
          split, budget, seed=args.seed, epoch=0, shuffle_shards=split == "train",
          gcs_root=args.gcs_root, total_shards=args.total_shards)
      inspected["selected_shard_paths"] = selected
      inspected["path"] = selected[0]
      inspect_record(selected[0], inspected)
      for key, values in inspected["metadata"].items():
        if key.rsplit("/", 1)[-1].lower() not in _ID_KEYS or not values:
          continue
        identity = (key, json.dumps(values, sort_keys=True))
        if identity in seen_ids:
          raise ValueError(
              f"duplicate inspected sample ID in {key!r}: "
              f"also present in {seen_ids[identity]!r} split")
        seen_ids[identity] = split
      inspected["status"] = "passed"
    except Exception as error:
      inspected["error"] = {"type": type(error).__name__, "message": str(error)}
      report["errors"].append({
          "split": split, "path": inspected.get("path"), **inspected["error"]})
      return report
  report["status"] = "passed"
  return report


def main() -> None:
  args = parse_args()
  report = run_preflight(args)
  os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
  with open(args.output, "w") as output:
    json.dump(report, output, indent=2, allow_nan=False)
  print(f"data preflight {report['status']}: {args.output}")
  if report["status"] != "passed":
    raise SystemExit(1)


if __name__ == "__main__":
  main()
