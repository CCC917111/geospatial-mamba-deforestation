#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the ForTy v1 test split.

Reports the metrics the ForTy benchmark uses: macro F1 over all classes
(Overall), the mean F1 of the three forest types (Forests) and the individual
F1 of natural forest (N), planted forest (P) and tree crops (TC).

Example:
    python -m mamba_forest.evaluate \
        --weights runs/mamba_forest/best.weights.h5 --test-shards 64
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import tensorflow as tf

from mamba_forest import data as data_lib
from mamba_forest.checkpoint import add_model_arguments, resolve_model_arguments
from mamba_forest.model import MambaForestSegmenter
from mamba_forest.train import evaluate, forest_scores


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--weights", required=True, help="Path to a .weights.h5 file")
  p.add_argument("--data-root", "--gcs-root", dest="gcs_root",
                 default=data_lib.GCS_ROOT,
                 help="Local directory or GCS root of the TFRecord shards")
  p.add_argument("--total-shards", type=int, default=data_lib.TOTAL_SHARDS,
                 help="Total shards per split, including the filename suffix")
  p.add_argument("--test-shards", type=int, default=64)
  p.add_argument("--batch-size", type=int, default=16)
  add_model_arguments(p)
  p.add_argument("--output-dir", default=None,
                 help="Where to write test_metrics.json "
                      "(defaults to the directory of the weights)")
  args = p.parse_args()
  try:
    return resolve_model_arguments(args)
  except (ValueError, OSError) as error:
    p.error(str(error))


def weights_sha256(path: str) -> str:
  """Identifies a local checkpoint without loading its bytes into memory."""
  digest = hashlib.sha256()
  with open(path, "rb") as checkpoint:
    for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def main() -> None:
  args = parse_args()
  num_classes = data_lib.NUM_CLASSES
  output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.weights))
  os.makedirs(output_dir, exist_ok=True)

  test_ds = data_lib.make_dataset(
      "test", args.test_shards, batch_size=args.batch_size,
      training=False, gcs_root=args.gcs_root, total_shards=args.total_shards)

  model = MambaForestSegmenter(
      num_classes=num_classes, d_model=args.d_model, dropout_rate=args.dropout,
      d_state_temporal=args.d_state_temporal,
      d_state_spatial=args.d_state_spatial,
      bottleneck_channels=args.bottleneck_channels)
  # Build the variables before loading the weights.
  built = False
  for batch in test_ds.take(1):
    inputs, _ = data_lib.split_inputs_labels(batch)
    model(inputs, training=False)
    built = True
  if not built:
    raise ValueError(f"test split under {args.gcs_root} yielded no batches")
  checkpoint_sha256 = weights_sha256(args.weights)
  model.load_weights(args.weights)

  accuracy, mean_iou, per_class_f1, confusion = evaluate(
      model, test_ds, num_classes, return_confusion=True)
  scores = forest_scores(per_class_f1)

  print(f"pixel accuracy : {accuracy:.4f}")
  print(f"mean IoU       : {mean_iou:.4f}")
  print(f"Overall (F1)   : {100 * scores['overall_f1']:.2f}")
  print(f"Forests (F1)   : {100 * scores['forests_f1']:.2f}")
  print(f"  N            : {100 * scores['natural_forest_f1']:.2f}")
  print(f"  P            : {100 * scores['planted_forest_f1']:.2f}")
  print(f"  TC           : {100 * scores['tree_crops_f1']:.2f}")
  print("per-class F1:")
  for name, value in zip(data_lib.CLASS_NAMES, per_class_f1):
    print(f"  {name:18s} {value:.4f}")

  metrics = {
      "weights": args.weights,
      "weights_sha256": checkpoint_sha256,
      "training_config_path": getattr(args, "training_config_path", None),
      "gcs_root": args.gcs_root,
      "split": "test",
      "test_shards": args.test_shards,
      "total_shards_per_split": args.total_shards,
      "shard_paths": data_lib.selected_shard_paths(
          "test", args.test_shards, gcs_root=args.gcs_root,
          total_shards=args.total_shards),
      "batch_size": args.batch_size,
      "model_config": model.get_config(),
      "runtime": {
          "tensorflow_version": tf.__version__,
          "precision_policy": tf.keras.mixed_precision.global_policy().name,
          "devices": [
              {"name": device.name, "type": device.device_type}
              for device in tf.config.list_physical_devices()
          ],
      },
      "num_pixels": int(confusion.sum()),
      "confusion_matrix": confusion.tolist(),
      "confusion_axes": {"rows": "true", "columns": "predicted"},
      "class_support": {
          name: int(count)
          for name, count in zip(data_lib.CLASS_NAMES, confusion.sum(axis=1))
      },
      "metric_conventions": {
          "overall_f1": "macro mean over all classes; absent classes have F1 0",
          "mean_iou": "mean over classes with nonzero true or predicted support",
      },
      "pixel_accuracy": accuracy,
      "mean_iou": mean_iou,
      **scores,
      "per_class_f1": {
          name: float(value)
          for name, value in zip(data_lib.CLASS_NAMES, per_class_f1)
      },
  }
  metrics_path = os.path.join(output_dir, "test_metrics.json")
  with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
  print(f"metrics written to {metrics_path}")


if __name__ == "__main__":
  main()
