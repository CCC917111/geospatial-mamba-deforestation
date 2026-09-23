#!/usr/bin/env python3
"""Train the CNN-Mamba forest segmenter on ForTy v1.

Example:
    python -m mamba_forest.train \
        --train-shards 128 --val-shards 8 --epochs 40 \
        --output-dir runs/mamba_forest

Each epoch reads a fresh random subset of training shards and a fixed subset of
validation shards, so validation metrics are comparable across epochs. The
learning rate follows a cosine decay from its initial value down to 10% of it,
per-epoch metrics are written to ``training_results.csv`` in the output
directory, and the checkpoint with the best macro F1 is kept.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import tensorflow as tf

from mamba_forest import data as data_lib
from mamba_forest.losses import DEFAULT_CLASS_WEIGHTS, ComboLoss
from mamba_forest.model import MambaForestSegmenter

BASE_COLUMNS = [
    "epoch", "train_loss", "val_accuracy", "val_mean_iou",
    "overall_f1", "forests_f1",
]


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--output-dir", default="runs/mamba_forest",
                 help="Directory for checkpoints, metrics and the config dump")
  p.add_argument("--gcs-root", default=data_lib.GCS_ROOT,
                 help="Root of the ForTy v1 TFRecord shards")
  p.add_argument("--train-shards", type=int, default=128,
                 help="Training shards read per epoch (1024 = full split)")
  p.add_argument("--val-shards", type=int, default=8,
                 help="Validation shards (fixed across epochs)")
  p.add_argument("--batch-size", type=int, default=16)
  p.add_argument("--epochs", type=int, default=40)
  p.add_argument("--learning-rate", type=float, default=2e-4,
                 help="Initial learning rate of the cosine schedule")
  p.add_argument("--lr-alpha", type=float, default=0.1,
                 help="Final learning rate as a fraction of the initial one")
  p.add_argument("--weight-decay", type=float, default=5e-3)
  p.add_argument("--d-model", type=int, default=32,
                 help="Channel width after modality projection and fusion")
  p.add_argument("--dropout", type=float, default=0.2)
  p.add_argument("--d-state-temporal", type=int, default=8,
                 help="State dimension of the temporal Mamba block")
  p.add_argument("--d-state-spatial", type=int, default=16,
                 help="State dimension of the spatial Mamba block")
  p.add_argument("--bottleneck-channels", type=int, default=512,
                 help="Channel width of the bottleneck feature map")
  p.add_argument("--combo-alpha", type=float, default=0.4,
                 help="Weight of the cross-entropy term against the Dice term")
  p.add_argument("--label-smoothing", type=float, default=0.05)
  p.add_argument("--no-class-weights", action="store_true",
                 help="Use uniform class weights in the cross-entropy term")
  p.add_argument("--grad-clip", type=float, default=1.0,
                 help="Clip gradients to this global norm; 0 disables clipping")
  p.add_argument("--no-mixed-precision", action="store_true",
                 help="Train in float32 instead of mixed float16")
  p.add_argument("--patience", type=int, default=8,
                 help="Early stopping patience in epochs (0 disables it)")
  p.add_argument("--seed", type=int, default=42)
  return p.parse_args()


def build_datasets(args):
  train_ds = data_lib.make_dataset(
      "train", args.train_shards, batch_size=args.batch_size,
      training=True, seed=args.seed, gcs_root=args.gcs_root)
  val_ds = data_lib.make_dataset(
      "validation", args.val_shards, batch_size=args.batch_size,
      training=False, gcs_root=args.gcs_root)
  return train_ds, val_ds


def cosine_learning_rate(initial: float, alpha: float, epoch: int,
                         total_epochs: int) -> float:
  """Cosine decay from ``initial`` down to ``alpha * initial``."""
  if total_epochs <= 1:
    return initial
  progress = min(max(epoch / float(total_epochs - 1), 0.0), 1.0)
  decayed = 0.5 * (1.0 + math.cos(math.pi * progress))
  return initial * (alpha + (1.0 - alpha) * decayed)


def _uses_keras2_loss_scaling(optimizer) -> bool:
  """True for the Keras 2 loss-scaling API, which unscales explicitly."""
  return hasattr(optimizer, "get_scaled_loss") and hasattr(
      optimizer, "get_unscaled_gradients")


def scale_loss(optimizer, loss):
  """Scales the loss when the optimizer performs dynamic loss scaling."""
  if _uses_keras2_loss_scaling(optimizer):
    return optimizer.get_scaled_loss(loss)
  if hasattr(optimizer, "scale_loss"):
    return optimizer.scale_loss(loss)
  return loss


def unscale_gradients(optimizer, gradients):
  """Undoes the loss scaling where the optimizer does not do it itself.

  Keras 2 expects the caller to unscale before ``apply_gradients``; Keras 3
  unscales inside it, so the two branches must not both run.
  """
  if _uses_keras2_loss_scaling(optimizer):
    return optimizer.get_unscaled_gradients(gradients)
  return gradients


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

  The confusion matrix is accumulated batch by batch, so evaluating many shards
  does not require holding every pixel label in memory.
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


def forest_scores(per_class_f1: np.ndarray) -> dict:
  """Aggregates the per-class F1 into the metrics the benchmark reports."""
  indices = list(data_lib.FOREST_CLASS_INDICES)
  return {
      "overall_f1": float(np.mean(per_class_f1)),
      "forests_f1": float(np.mean(per_class_f1[indices])),
      "natural_forest_f1": float(per_class_f1[data_lib.NATURAL_FOREST_INDEX]),
      "planted_forest_f1": float(per_class_f1[data_lib.PLANTED_FOREST_INDEX]),
      "tree_crops_f1": float(per_class_f1[data_lib.TREE_CROPS_INDEX]),
  }


def main() -> None:
  args = parse_args()
  tf.random.set_seed(args.seed)
  np.random.seed(args.seed)
  os.makedirs(args.output_dir, exist_ok=True)
  with open(os.path.join(args.output_dir, "config.json"), "w") as f:
    json.dump(vars(args), f, indent=2, default=str)

  if not args.no_mixed_precision:
    # Half precision for the compute-heavy operations, float32 for the weights
    # and the softmax head, with dynamic loss scaling to prevent underflow.
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

  num_classes = data_lib.NUM_CLASSES
  train_ds, val_ds = build_datasets(args)

  model = MambaForestSegmenter(
      num_classes=num_classes, d_model=args.d_model, dropout_rate=args.dropout,
      d_state_temporal=args.d_state_temporal,
      d_state_spatial=args.d_state_spatial,
      bottleneck_channels=args.bottleneck_channels)
  loss_fn = ComboLoss(
      num_classes, alpha=args.combo_alpha,
      label_smoothing=args.label_smoothing,
      class_weights=None if args.no_class_weights else DEFAULT_CLASS_WEIGHTS)

  base_optimizer = tf.keras.optimizers.AdamW(
      learning_rate=args.learning_rate, weight_decay=args.weight_decay,
      global_clipnorm=args.grad_clip if args.grad_clip > 0 else None)
  optimizer = base_optimizer
  if not args.no_mixed_precision:
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)

  train_loss = tf.keras.metrics.Mean()

  # Build the model's weights on a real batch before the training step is
  # traced, so that the shapes are settled and a data or credential problem
  # surfaces here rather than inside the traced function.
  built = False
  for batch in train_ds.take(1):
    warmup_inputs, _ = data_lib.split_inputs_labels(batch)
    model(warmup_inputs, training=False)
    built = True
  if not built:
    raise RuntimeError(
        f"the training split under {args.gcs_root} yielded no batches; check "
        "the dataset root and the Google Cloud credentials")
  print(f"model built with {model.count_params():,} parameters")

  @tf.function
  def train_step(inputs, labels):
    with tf.GradientTape() as tape:
      predictions = model(inputs, training=True)
      loss = loss_fn(labels, predictions)
      scaled = scale_loss(optimizer, loss)
    gradients = tape.gradient(scaled, model.trainable_variables)
    gradients = unscale_gradients(optimizer, gradients)
    pairs = [(g, v) for g, v in zip(gradients, model.trainable_variables)
             if g is not None]
    optimizer.apply_gradients(pairs)
    train_loss.update_state(loss)

  history, best_score, epochs_without_improvement = [], -1.0, 0
  results_path = os.path.join(args.output_dir, "training_results.csv")
  columns = BASE_COLUMNS + [
      "f1_" + name.lower().replace(" ", "_") for name in data_lib.CLASS_NAMES]

  for epoch in range(1, args.epochs + 1):
    learning_rate = cosine_learning_rate(
        args.learning_rate, args.lr_alpha, epoch - 1, args.epochs)
    base_optimizer.learning_rate.assign(learning_rate)

    train_loss.reset_state()
    for batch in train_ds:
      inputs, labels = data_lib.split_inputs_labels(batch)
      train_step(inputs, labels)

    accuracy, mean_iou, per_class_f1 = evaluate(model, val_ds, num_classes)
    scores = forest_scores(per_class_f1)

    row = ([epoch, float(train_loss.result()), accuracy, mean_iou,
            scores["overall_f1"], scores["forests_f1"]]
           + [float(v) for v in per_class_f1])
    history.append(row)
    print(f"epoch {epoch:3d}/{args.epochs} | lr {learning_rate:.2e} | "
          f"loss {row[1]:.4f} | val acc {accuracy:.4f} | mIoU {mean_iou:.4f} | "
          f"overall F1 {scores['overall_f1']:.4f} | "
          f"forests F1 {scores['forests_f1']:.4f}")
    print(f"  N {scores['natural_forest_f1']:.4f} | "
          f"P {scores['planted_forest_f1']:.4f} | "
          f"TC {scores['tree_crops_f1']:.4f}")

    with open(results_path, "w") as f:
      f.write(",".join(columns) + "\n")
      for r in history:
        f.write(f"{r[0]}," + ",".join(f"{v:.6f}" for v in r[1:]) + "\n")

    model.save_weights(os.path.join(args.output_dir, "last.weights.h5"))
    if scores["overall_f1"] > best_score:
      best_score = scores["overall_f1"]
      epochs_without_improvement = 0
      model.save_weights(os.path.join(args.output_dir, "best.weights.h5"))
      print(f"  new best overall F1: {best_score:.4f}")
    else:
      epochs_without_improvement += 1
      if args.patience and epochs_without_improvement >= args.patience:
        print(f"early stopping after {epoch} epochs "
              f"({args.patience} without improvement)")
        break

  print(f"best overall F1: {best_score:.4f}")
  print(f"metrics written to {results_path}")


if __name__ == "__main__":
  main()
