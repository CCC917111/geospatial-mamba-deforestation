#!/usr/bin/env python3
"""Train the Mamba forest segmenter on ForTy v1.

Example:
    python -m mamba_forest.train \
        --train-shards 32 --val-shards 8 --epochs 20 \
        --output-dir runs/mamba_exp06

Each epoch reads a fresh random subset of training shards and a fixed subset of
validation shards, so validation metrics are comparable across epochs. Per-epoch
metrics are appended to ``training_results.csv`` in the output directory and the
best checkpoint (by macro F1) is kept.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import tensorflow as tf

from mamba_forest import data as data_lib
from mamba_forest.losses import ComboLoss
from mamba_forest.model import MambaForestSegmenter

BASE_COLUMNS = [
    "epoch", "train_loss", "val_accuracy", "val_mean_iou",
    "macro_f1", "forest_f1",
]


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--output-dir", default="runs/mamba_forest",
                 help="Directory for checkpoints, metrics and the config dump")
  p.add_argument("--gcs-root", default=data_lib.GCS_ROOT,
                 help="Root of the ForTy v1 TFRecord shards")
  p.add_argument("--train-shards", type=int, default=32,
                 help="Training shards read per epoch (1024 = full split)")
  p.add_argument("--val-shards", type=int, default=8,
                 help="Validation shards (fixed across epochs)")
  p.add_argument("--batch-size", type=int, default=8)
  p.add_argument("--epochs", type=int, default=20)
  p.add_argument("--learning-rate", type=float, default=1e-4)
  p.add_argument("--weight-decay", type=float, default=1e-4)
  p.add_argument("--d-model", type=int, default=32,
                 help="Channel width after modality projection and fusion")
  p.add_argument("--dropout", type=float, default=0.2)
  p.add_argument("--loss", choices=["cce", "combo"], default="cce",
                 help="Plain cross-entropy or the cross-entropy + Dice combination")
  p.add_argument("--combo-alpha", type=float, default=0.5,
                 help="Weight of the cross-entropy term when --loss=combo")
  p.add_argument("--grad-clip", type=float, default=1.0,
                 help="Clip gradients to [-v, v]; 0 disables clipping")
  p.add_argument("--patience", type=int, default=5,
                 help="Early stopping patience in epochs (0 disables it)")
  p.add_argument("--seed", type=int, default=42)
  return p.parse_args()


def build_datasets(args):
  train_ds = data_lib.make_dataset(
      "train", args.train_shards, batch_size=args.batch_size,
      shuffle_shards=True, seed=args.seed, gcs_root=args.gcs_root)
  val_ds = data_lib.make_dataset(
      "validation", args.val_shards, batch_size=args.batch_size,
      shuffle_shards=False, gcs_root=args.gcs_root)
  return train_ds, val_ds


def f1_from_confusion(confusion: np.ndarray) -> np.ndarray:
  """Per-class F1 from a confusion matrix with rows = true, columns = predicted."""
  true_positives = np.diag(confusion).astype(np.float64)
  predicted = confusion.sum(axis=0)
  actual = confusion.sum(axis=1)
  denominator = predicted + actual
  return np.divide(2.0 * true_positives, denominator,
                   out=np.zeros_like(true_positives),
                   where=denominator > 0)


def evaluate(model, dataset, num_classes: int):
  """Returns (accuracy, mean IoU, per-class F1) on ``dataset``.

  The confusion matrix is accumulated batch by batch, so evaluating many
  shards does not require holding every pixel label in memory.
  """
  accuracy = tf.keras.metrics.CategoricalAccuracy()
  mean_iou = tf.keras.metrics.MeanIoU(num_classes)
  confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

  for batch in dataset:
    inputs, labels = data_lib.split_inputs_labels(batch)
    predictions = model(inputs, training=False)
    accuracy.update_state(labels, predictions)
    y_true = tf.argmax(labels, axis=-1)
    y_pred = tf.argmax(predictions, axis=-1)
    mean_iou.update_state(y_true, y_pred)
    confusion += tf.math.confusion_matrix(
        tf.reshape(y_true, [-1]), tf.reshape(y_pred, [-1]),
        num_classes=num_classes, dtype=tf.int64).numpy()

  return float(accuracy.result()), float(mean_iou.result()), f1_from_confusion(confusion)


def main() -> None:
  args = parse_args()
  tf.random.set_seed(args.seed)
  np.random.seed(args.seed)
  os.makedirs(args.output_dir, exist_ok=True)
  with open(os.path.join(args.output_dir, "config.json"), "w") as f:
    json.dump(vars(args), f, indent=2)

  num_classes = data_lib.NUM_CLASSES
  train_ds, val_ds = build_datasets(args)

  model = MambaForestSegmenter(
      num_classes=num_classes, d_model=args.d_model, dropout_rate=args.dropout)
  if args.loss == "combo":
    loss_fn = ComboLoss(num_classes, alpha=args.combo_alpha)
  else:
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
  optimizer = tf.keras.optimizers.AdamW(
      learning_rate=args.learning_rate, weight_decay=args.weight_decay)

  train_loss = tf.keras.metrics.Mean()

  @tf.function
  def train_step(inputs, labels):
    with tf.GradientTape() as tape:
      predictions = model(inputs, training=True)
      loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    pairs = [(g, v) for g, v in zip(gradients, model.trainable_variables)
             if g is not None]
    if args.grad_clip > 0:
      pairs = [(tf.clip_by_value(g, -args.grad_clip, args.grad_clip), v)
               for g, v in pairs]
    optimizer.apply_gradients(pairs)
    train_loss.update_state(loss)

  history, best_score, epochs_without_improvement = [], -1.0, 0
  results_path = os.path.join(args.output_dir, "training_results.csv")
  forest_indices = list(data_lib.FOREST_CLASS_INDICES)
  columns = BASE_COLUMNS + [f"f1_class_{i}" for i in range(num_classes)]

  for epoch in range(1, args.epochs + 1):
    train_loss.reset_state()
    for batch in train_ds:
      inputs, labels = data_lib.split_inputs_labels(batch)
      train_step(inputs, labels)

    accuracy, mean_iou, per_class_f1 = evaluate(model, val_ds, num_classes)
    macro_f1 = float(np.mean(per_class_f1))
    forest_f1 = float(np.mean(per_class_f1[forest_indices]))

    row = ([epoch, float(train_loss.result()), accuracy, mean_iou,
            macro_f1, forest_f1] + [float(v) for v in per_class_f1])
    history.append(row)
    print(f"epoch {epoch:3d}/{args.epochs} | loss {row[1]:.4f} | "
          f"val acc {accuracy:.4f} | mIoU {mean_iou:.4f} | "
          f"macro F1 {macro_f1:.4f} | forest F1 {forest_f1:.4f}")
    print("  per-class F1: " + ", ".join(
        f"{name}={value:.3f}"
        for name, value in zip(data_lib.CLASS_NAMES, per_class_f1)))

    with open(results_path, "w") as f:
      f.write(",".join(columns) + "\n")
      for r in history:
        f.write(f"{r[0]}," + ",".join(f"{v:.6f}" for v in r[1:]) + "\n")

    model.save_weights(os.path.join(args.output_dir, "last.weights.h5"))
    if macro_f1 > best_score:
      best_score = macro_f1
      epochs_without_improvement = 0
      model.save_weights(os.path.join(args.output_dir, "best.weights.h5"))
      print(f"  new best macro F1: {best_score:.4f}")
    else:
      epochs_without_improvement += 1
      if args.patience and epochs_without_improvement >= args.patience:
        print(f"early stopping after {epoch} epochs "
              f"({args.patience} without improvement)")
        break

  print(f"best macro F1: {best_score:.4f}")
  print(f"metrics written to {results_path}")


if __name__ == "__main__":
  main()
