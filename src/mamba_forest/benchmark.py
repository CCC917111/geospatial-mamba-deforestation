"""Measure synthetic model compute with warmup and output synchronization.

Example:
    python -m mamba_forest.benchmark --mode both --tile-size 32 \
        --batch-size 1 --d-model 8 --bottleneck-channels 32 \
        --steps 10 --output results/performance_cpu.json

This measures normalized synthetic tensors already in memory. It excludes
dataset IO, preprocessing, checkpoint loading, graph tracing and output-file
writes. These timings say nothing about trained accuracy or full-job runtime.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from mamba_forest import data as data_lib
from mamba_forest.constants import DEFAULT_MODEL_VARIANT, MODEL_VARIANTS
from mamba_forest.losses import ComboLoss
from mamba_forest.model import MODEL_DEFAULTS, MambaForestSegmenter
from mamba_forest.train import scale_loss, unscale_gradients


def parse_args(argv=None) -> argparse.Namespace:
  """Parses the compute benchmark settings; no dataset is required."""
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--mode", choices=("inference", "training", "both"),
                      default="inference")
  parser.add_argument("--inference-mode", choices=("eager", "graph", "both"),
                      default="both")
  parser.add_argument("--precision", choices=("float32", "mixed_float16"),
                      default="float32")
  parser.add_argument("--device", help="Optional TensorFlow device, e.g. /CPU:0")
  parser.add_argument("--tile-size", type=int, default=128)
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--steps", type=int, default=10)
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--d-model", type=int, default=MODEL_DEFAULTS["d_model"])
  parser.add_argument("--d-state-temporal", type=int,
                      default=MODEL_DEFAULTS["d_state_temporal"])
  parser.add_argument("--d-state-spatial", type=int,
                      default=MODEL_DEFAULTS["d_state_spatial"])
  parser.add_argument("--bottleneck-channels", type=int,
                      default=MODEL_DEFAULTS["bottleneck_channels"])
  parser.add_argument("--dropout", type=float, default=MODEL_DEFAULTS["dropout_rate"])
  parser.add_argument("--model-variant", choices=MODEL_VARIANTS,
                      default=DEFAULT_MODEL_VARIANT)
  parser.add_argument("--output", type=Path, help="Optional JSON output path")
  args = parser.parse_args(argv)
  for name in ("tile_size", "batch_size", "steps", "warmup", "d_model",
               "d_state_temporal", "d_state_spatial", "bottleneck_channels"):
    if getattr(args, name) < 1:
      parser.error(f"--{name.replace('_', '-')} must be positive")
  if args.tile_size % 8:
    parser.error("--tile-size must be divisible by 8")
  if not 0 <= args.dropout < 1:
    parser.error("--dropout must be in [0, 1)")
  return args


def summarize_latencies(latencies, batch_size: int) -> dict:
  """Summarizes synchronized seconds per step without discarding samples."""
  values = np.asarray(latencies, dtype=np.float64)
  invalid = (values.ndim != 1 or values.size == 0 or not np.isfinite(values).all()
             or np.any(values <= 0) or batch_size < 1)
  if invalid:
    raise ValueError("positive finite latencies and batch_size are required")
  return {
      "latencies_seconds": values.tolist(),
      "median_seconds": float(np.median(values)),
      "p95_seconds": float(np.percentile(values, 95)),
      "tiles_per_second": float(batch_size * values.size / values.sum()),
      "measured_steps": int(values.size),
  }


def measure(step, steps: int, warmup: int, batch_size: int) -> dict:
  """Warms up, then times completed steps using a host-read barrier."""
  if steps < 1 or warmup < 1:
    raise ValueError("steps and warmup must be positive")

  def synchronized_step():
    for value in tf.nest.flatten(step()):
      if not np.isfinite(value.numpy()).all():
        raise FloatingPointError("benchmark step produced non-finite outputs")

  for _ in range(warmup):
    synchronized_step()
  latencies = []
  for _ in range(steps):
    start = time.perf_counter()
    synchronized_step()
    latencies.append(time.perf_counter() - start)
  return summarize_latencies(latencies, batch_size)


def synthetic_batch(batch_size: int, tile_size: int):
  """Returns fixed-seed normalized inputs and one-hot pixel labels."""
  shapes = data_lib.FEATURE_SHAPES
  inputs = {
      "s2": tf.random.uniform(
          [batch_size, shapes["s2"][0], tile_size, tile_size, shapes["s2"][-1]]),
      "elevation": tf.random.normal(
          [batch_size, tile_size, tile_size, shapes["elevation"][-1]]),
      "climate": tf.random.normal([batch_size, *shapes["climate"]]),
  }
  labels = tf.one_hot(tf.random.uniform(
      [batch_size, tile_size, tile_size], maxval=data_lib.NUM_CLASSES,
      dtype=tf.int32), data_lib.NUM_CLASSES)
  return inputs, labels


def run_benchmark(args) -> dict:
  """Runs independent inference and training models and returns metadata."""
  previous_policy = tf.keras.mixed_precision.global_policy()
  tf.keras.mixed_precision.set_global_policy(args.precision)
  config = {
      "num_classes": data_lib.NUM_CLASSES, "d_model": args.d_model,
      "dropout_rate": args.dropout, "d_state_temporal": args.d_state_temporal,
      "d_state_spatial": args.d_state_spatial,
      "bottleneck_channels": args.bottleneck_channels,
      "model_variant": getattr(args, "model_variant", DEFAULT_MODEL_VARIANT),
  }
  results = {}
  try:
    context = tf.device(args.device) if args.device else contextlib.nullcontext()
    with context:
      tf.keras.utils.set_random_seed(args.seed)
      inputs, labels = synthetic_batch(args.batch_size, args.tile_size)
      model = MambaForestSegmenter(**config)
      model(inputs, training=False)
      metadata = {
          "tensorflow_version": tf.__version__,
          "python_version": platform.python_version(),
          "platform": platform.platform(),
          "cpu_count": os.cpu_count(),
          "thread_settings": {
              name: os.environ.get(name) for name in (
                  "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS",
                  "OMP_NUM_THREADS")},
          "tensorflow_threading": {
              "intra_op_parallelism_threads":
                  tf.config.threading.get_intra_op_parallelism_threads(),
              "inter_op_parallelism_threads":
                  tf.config.threading.get_inter_op_parallelism_threads(),
          },
          "tensorflow_threading_note": (
              "0 indicates no explicit tf.config.threading override; "
              "environment settings are recorded separately."),
          "logical_devices": [device.name for device in
                              tf.config.list_logical_devices()],
          "requested_device": args.device,
          "input_devices": {key: value.device for key, value in inputs.items()},
          "precision": args.precision, "seed": args.seed,
          "input_shapes": {key: value.shape.as_list() for key, value in inputs.items()},
          "label_shape": labels.shape.as_list(),
          "model_config": config, "parameters": model.count_params(),
          "warmup_steps": args.warmup, "steps": args.steps,
          "scope": ("Synthetic model compute plus scalar output synchronization; "
                    "excludes dataset IO, preprocessing, checkpoint loading, "
                    "graph tracing and output-file writes. No accuracy measurement."),
      }

      if args.mode in ("inference", "both"):
        def infer():
          # Scalar reduction forces completion without copying a probability map.
          return tf.reduce_sum(model(inputs, training=False))

        modes = ("eager", "graph") if args.inference_mode == "both" else (
            args.inference_mode,)
        for mode in modes:
          step = infer if mode == "eager" else tf.function(infer)
          results[f"inference_{mode}"] = measure(
              step, args.steps, args.warmup, args.batch_size)

      if args.mode in ("training", "both"):
        # A fresh model prevents training measurements from changing the model
        # used for the eager-versus-graph inference comparison.
        tf.keras.utils.set_random_seed(args.seed)
        train_model = MambaForestSegmenter(**config)
        train_model(inputs, training=False)
        loss_fn = ComboLoss(data_lib.NUM_CLASSES)
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=2e-4, weight_decay=5e-3, global_clipnorm=1.0)
        if args.precision == "mixed_float16":
          optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        @tf.function
        def train():
          with tf.GradientTape() as tape:
            predictions = train_model(inputs, training=True)
            loss = loss_fn(labels, predictions)
            scaled = scale_loss(optimizer, loss)
          gradients = tape.gradient(scaled, train_model.trainable_variables)
          gradients = unscale_gradients(optimizer, gradients)
          optimizer.apply_gradients(
              (gradient, variable) for gradient, variable in
              zip(gradients, train_model.trainable_variables) if gradient is not None)
          return loss, tf.identity(optimizer.iterations)

        results["training_graph"] = measure(
            train, args.steps, args.warmup, args.batch_size)
        metadata["training_optimizer"] = {
            "name": "AdamW", "learning_rate": 2e-4,
            "weight_decay": 5e-3, "global_clipnorm": 1.0,
            "dynamic_loss_scaling": args.precision == "mixed_float16",
        }
        metadata["training_loss"] = loss_fn.get_config()
  finally:
    tf.keras.mixed_precision.set_global_policy(previous_policy)
  return {"metadata": metadata, "results": results}


def main() -> None:
  args = parse_args()
  results = run_benchmark(args)
  payload = json.dumps(results, indent=2)
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
  print(payload)


if __name__ == "__main__":
  main()
