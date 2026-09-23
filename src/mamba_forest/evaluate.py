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
import json
import os

import numpy as np

from mamba_forest import data as data_lib
from mamba_forest.model import MambaForestSegmenter
from mamba_forest.train import evaluate, forest_scores


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--weights", required=True, help="Path to a .weights.h5 file")
  p.add_argument("--gcs-root", default=data_lib.GCS_ROOT,
                 help="Root of the ForTy v1 TFRecord shards")
  p.add_argument("--test-shards", type=int, default=64)
  p.add_argument("--batch-size", type=int, default=16)
  p.add_argument("--d-model", type=int, default=32,
                 help="Must match the value used for training")
  p.add_argument("--dropout", type=float, default=0.2)
  p.add_argument("--d-state-temporal", type=int, default=8,
                 help="Must match the value used for training")
  p.add_argument("--d-state-spatial", type=int, default=16,
                 help="Must match the value used for training")
  p.add_argument("--bottleneck-channels", type=int, default=512,
                 help="Must match the value used for training")
  p.add_argument("--output-dir", default=None,
                 help="Where to write test_metrics.json "
                      "(defaults to the directory of the weights)")
  return p.parse_args()


def main() -> None:
  args = parse_args()
  num_classes = data_lib.NUM_CLASSES
  output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.weights))
  os.makedirs(output_dir, exist_ok=True)

  test_ds = data_lib.make_dataset(
      "test", args.test_shards, batch_size=args.batch_size,
      training=False, gcs_root=args.gcs_root)

  model = MambaForestSegmenter(
      num_classes=num_classes, d_model=args.d_model, dropout_rate=args.dropout,
      d_state_temporal=args.d_state_temporal,
      d_state_spatial=args.d_state_spatial,
      bottleneck_channels=args.bottleneck_channels)
  # Build the variables before loading the weights.
  for batch in test_ds.take(1):
    inputs, _ = data_lib.split_inputs_labels(batch)
    model(inputs, training=False)
  model.load_weights(args.weights)

  accuracy, mean_iou, per_class_f1 = evaluate(model, test_ds, num_classes)
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
      "test_shards": args.test_shards,
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
