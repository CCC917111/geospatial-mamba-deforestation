#!/usr/bin/env python3
"""Train the CNN-Mamba forest segmenter on ForTy v1.

Example:
    python -m mamba_forest.train \
        --train-shards 128 --val-shards 8 --epochs 40 \
        --output-dir runs/mamba_forest

Each epoch reads an explicit seeded subset of training shards and a fixed
validation subset. Training membership rotates unless --fixed-train-subset is
set; exact paths are recorded in split_manifest.json. The
learning rate follows a cosine decay from its initial value down to 10% of it,
per-epoch metrics are written to ``training_results.csv`` in the output
directory, and the checkpoint with the best macro F1 is kept.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import tensorflow as tf

from mamba_forest import data as data_lib
from mamba_forest.losses import DEFAULT_CLASS_WEIGHTS, ComboLoss
from mamba_forest.model import MODEL_DEFAULTS, MambaForestSegmenter

BASE_COLUMNS = [
    "epoch", "train_loss", "val_accuracy", "val_mean_iou",
    "overall_f1", "forests_f1",
]
TIMING_COLUMNS = [
    "train_seconds", "validation_seconds", "train_examples",
    "train_examples_per_second",
]


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--output-dir", default="runs/mamba_forest",
                 help="Directory for checkpoints, metrics and the config dump")
  p.add_argument("--data-root", "--gcs-root", dest="gcs_root",
                 default=data_lib.GCS_ROOT,
                 help="Local directory or GCS root of the TFRecord shards")
  p.add_argument("--total-shards", type=int, default=data_lib.TOTAL_SHARDS,
                 help="Total shards per split, including the filename suffix")
  p.add_argument("--train-shards", type=int, default=128,
                 help="Training shards read per epoch (total-shards = full split)")
  p.add_argument("--fixed-train-subset", action="store_true",
                 help="Reuse the same seeded training shards in every epoch")
  p.add_argument("--val-shards", type=int, default=8,
                 help="Validation shards (fixed across epochs)")
  p.add_argument("--batch-size", type=int, default=16)
  p.add_argument("--epochs", type=int, default=40)
  p.add_argument("--learning-rate", type=float, default=2e-4,
                 help="Initial learning rate of the cosine schedule")
  p.add_argument("--lr-alpha", type=float, default=0.1,
                 help="Final learning rate as a fraction of the initial one")
  p.add_argument("--weight-decay", type=float, default=5e-3)
  p.add_argument("--d-model", type=int, default=MODEL_DEFAULTS["d_model"],
                 help="Channel width after modality projection and fusion")
  p.add_argument("--dropout", type=float, default=MODEL_DEFAULTS["dropout_rate"])
  p.add_argument("--d-state-temporal", type=int,
                 default=MODEL_DEFAULTS["d_state_temporal"],
                 help="State dimension of the temporal Mamba block")
  p.add_argument("--d-state-spatial", type=int,
                 default=MODEL_DEFAULTS["d_state_spatial"],
                 help="State dimension of the spatial Mamba block")
  p.add_argument("--bottleneck-channels", type=int,
                 default=MODEL_DEFAULTS["bottleneck_channels"],
                 help="Channel width of the bottleneck feature map")
  p.add_argument("--combo-alpha", type=float, default=0.4,
                 help="Weight of the cross-entropy term against the Dice term")
  p.add_argument("--label-smoothing", type=float, default=0.05)
  p.add_argument("--no-class-weights", action="store_true",
                 help="Use uniform class weights in the cross-entropy term")
  p.add_argument("--grad-clip", type=float, default=1.0,
                 help="Clip gradients to this global norm; 0 disables clipping")
  p.add_argument("--precision", choices=("auto", "float32", "mixed_float16"),
                 default="auto",
                 help="auto uses mixed float16 on GPU and float32 on CPU")
  p.add_argument("--no-mixed-precision", action="store_true",
                 help="Alias for --precision float32")
  p.add_argument("--patience", type=int, default=8,
                 help="Early stopping patience in epochs (0 disables it)")
  p.add_argument("--seed", type=int, default=42)
  args = p.parse_args()
  if args.no_mixed_precision and args.precision == "mixed_float16":
    p.error("--no-mixed-precision conflicts with --precision mixed_float16")
  return args


def configure_precision(precision: str = "auto",
                        no_mixed_precision: bool = False) -> str:
  """Resolves the compute policy, resets global state and returns its name.

  Auto selects float32 on CPU because half-precision kernels can be slower
  there. An explicit policy supports controlled hardware measurements.
  """
  if precision not in ("auto", "float32", "mixed_float16"):
    raise ValueError(f"unknown precision policy: {precision}")
  if no_mixed_precision:
    if precision == "mixed_float16":
      raise ValueError("no_mixed_precision conflicts with mixed_float16")
    precision = "float32"
  if precision == "auto":
    precision = ("mixed_float16" if tf.config.get_visible_devices("GPU")
                 else "float32")
  tf.keras.mixed_precision.set_global_policy(precision)
  return precision


def prepare_output_directory(output_dir: str) -> None:
  """Creates a fresh run directory without relabeling existing checkpoints."""
  directory = Path(output_dir)
  if any(directory.glob("*.weights.h5")):
    raise FileExistsError(
        f"{directory} already contains checkpoint weights; "
        "choose a new --output-dir to keep its config.json paired with its weights")
  directory.mkdir(parents=True, exist_ok=True)


def build_train_dataset(args, epoch: int = 0):
  """Builds the explicitly selected training subset for a zero-based epoch."""
  shard_epoch = 0 if args.fixed_train_subset else epoch
  return data_lib.make_dataset(
      "train", args.train_shards, batch_size=args.batch_size,
      training=True, seed=args.seed, gcs_root=args.gcs_root, epoch=shard_epoch,
      total_shards=args.total_shards)


def build_datasets(args):
  """Builds the first training epoch and the fixed validation subset."""
  train_ds = build_train_dataset(args)
  val_ds = data_lib.make_dataset(
      "validation", args.val_shards, batch_size=args.batch_size,
      training=False, gcs_root=args.gcs_root, total_shards=args.total_shards)
  return train_ds, val_ds


def training_epoch_manifest(args, epoch: int) -> dict:
  """Records exact paths for a one-based epoch before its iterator starts."""
  shard_epoch = 0 if args.fixed_train_subset else epoch - 1
  return {
      "epoch": epoch,
      "shard_epoch": shard_epoch,
      "training_completed": False,
      "paths": data_lib.selected_shard_paths(
          "train", args.train_shards, seed=args.seed, epoch=shard_epoch,
          shuffle_shards=True, gcs_root=args.gcs_root,
          total_shards=args.total_shards),
  }


def write_split_manifest(path: str, manifest: dict) -> None:
  """Atomically writes selection provenance and completed unique coverage."""
  completed = {shard for epoch in manifest["epochs"]
               if epoch["training_completed"] for shard in epoch["paths"]}
  manifest["completed_unique_train_shards"] = len(completed)
  with open(path + ".tmp", "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2)
  os.replace(path + ".tmp", path)


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


def evaluate(model, dataset, num_classes: int, return_confusion: bool = False):
  """Returns (accuracy, mean IoU, per-class F1) on ``dataset``.

  Inference and int64 confusion updates run in one graph per batch, and only
  the final matrix is copied to NumPy. Mean IoU excludes classes absent from
  both labels and predictions; macro F1 retains all ``num_classes`` classes,
  with zero F1 for absent classes. An empty or zero-pixel dataset is rejected.
  With ``return_confusion=True``, the matrix is returned as a fourth value.
  """
  if num_classes < 1:
    raise ValueError("num_classes must be positive")
  confusion = tf.Variable(
      tf.zeros((num_classes, num_classes), dtype=tf.int64), trainable=False)

  @tf.function(reduce_retracing=True)
  def accumulate(inputs, labels):
    predictions = model(inputs, training=False)
    y_true = tf.argmax(labels, axis=-1)
    y_pred = tf.argmax(predictions, axis=-1)
    confusion.assign_add(tf.math.confusion_matrix(
        tf.reshape(y_true, [-1]), tf.reshape(y_pred, [-1]),
        num_classes=num_classes, dtype=tf.int64))

  for batch in dataset:
    inputs, labels = data_lib.split_inputs_labels(batch)
    accumulate(inputs, labels)

  matrix = confusion.numpy()
  total_pixels = int(matrix.sum())
  if total_pixels == 0:
    raise ValueError("evaluation dataset contains no pixels")
  true_positives = np.diag(matrix).astype(np.float64)
  union = matrix.sum(axis=0) + matrix.sum(axis=1) - true_positives
  present = union > 0
  result = (float(true_positives.sum() / total_pixels),
            float(np.mean(true_positives[present] / union[present])),
            f1_from_confusion(matrix))
  return result + (matrix,) if return_confusion else result


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
  if args.epochs < 1:
    raise ValueError("epochs must be positive")
  prepare_output_directory(args.output_dir)
  tf.keras.utils.set_random_seed(args.seed)
  resolved_precision = configure_precision(args.precision, args.no_mixed_precision)
  with open(os.path.join(args.output_dir, "config.json"), "w") as f:
    json.dump({**vars(args), "resolved_precision": resolved_precision,
               "tensorflow_version": tf.__version__}, f, indent=2, default=str)

  num_classes = data_lib.NUM_CLASSES
  train_ds, val_ds = build_datasets(args)
  manifest_path = os.path.join(args.output_dir, "split_manifest.json")
  manifest = {
      "schema_version": 1,
      "selection_protocol": "python-random-seed-plus-epoch-v1",
      "gcs_root": args.gcs_root,
      "total_shards_per_split": args.total_shards,
      "seed": args.seed,
      "training_mode": "fixed" if args.fixed_train_subset else "rotating",
      "validation_paths": data_lib.selected_shard_paths(
          "validation", args.val_shards, gcs_root=args.gcs_root,
          total_shards=args.total_shards),
      "epochs": [],
  }
  write_split_manifest(manifest_path, manifest)

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
  if resolved_precision == "mixed_float16":
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
      "f1_" + name.lower().replace(" ", "_") for name in data_lib.CLASS_NAMES
  ] + TIMING_COLUMNS

  for epoch in range(1, args.epochs + 1):
    if epoch > 1:
      train_ds = build_train_dataset(args, epoch - 1)
    selection = training_epoch_manifest(args, epoch)
    manifest["epochs"].append(selection)
    write_split_manifest(manifest_path, manifest)
    learning_rate = cosine_learning_rate(
        args.learning_rate, args.lr_alpha, epoch - 1, args.epochs)
    base_optimizer.learning_rate.assign(learning_rate)

    train_loss.reset_state()
    train_examples = 0
    started = time.perf_counter()
    for batch in train_ds:
      inputs, labels = data_lib.split_inputs_labels(batch)
      train_step(inputs, labels)
      train_examples += int(labels.shape[0])
    loss_value = float(train_loss.result())  # Synchronize before stopping time.
    train_seconds = time.perf_counter() - started
    if not train_examples:
      raise ValueError("the training epoch yielded no examples")
    selection["training_completed"] = True
    write_split_manifest(manifest_path, manifest)

    started = time.perf_counter()
    accuracy, mean_iou, per_class_f1 = evaluate(model, val_ds, num_classes)
    validation_seconds = time.perf_counter() - started
    scores = forest_scores(per_class_f1)

    row = ([epoch, loss_value, accuracy, mean_iou,
            scores["overall_f1"], scores["forests_f1"]]
           + [float(v) for v in per_class_f1]
           + [train_seconds, validation_seconds, train_examples,
              train_examples / train_seconds])
    history.append(row)
    print(f"epoch {epoch:3d}/{args.epochs} | lr {learning_rate:.2e} | "
          f"loss {row[1]:.4f} | val acc {accuracy:.4f} | mIoU {mean_iou:.4f} | "
          f"overall F1 {scores['overall_f1']:.4f} | "
          f"forests F1 {scores['forests_f1']:.4f}")
    print(f"  train {train_seconds:.2f}s | val {validation_seconds:.2f}s | "
          f"{train_examples / train_seconds:.2f} training tiles/s")
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
