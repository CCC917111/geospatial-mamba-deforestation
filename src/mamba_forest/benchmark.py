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
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from mamba_forest import data as data_lib
from mamba_forest.checkpoint import weights_sha256
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
  parser.add_argument("--weights", type=Path,
                      help="Optional shared checkpoint; architecture flags must "
                           "match. Loading is excluded from measured steps")
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


def _memory_start(device):
  """Resets allocator peaks after warmup, retaining the live-memory baseline."""
  report = {"device": device, "status": "not_applicable",
            "scope": "TensorFlow allocator bytes; not total or reserved GPU VRAM"}
  if device is None:
    return report
  try:
    current = tf.config.experimental.get_memory_info(device)["current"]
    tf.config.experimental.reset_memory_stats(device)
    report.update({"status": "measured", "starting_current_bytes": int(current)})
  except (ValueError, RuntimeError) as error:
    report.update({"status": "unavailable", "reason": str(error)})
  return report


def _memory_finish(report):
  if report["status"] == "measured":
    try:
      observed = tf.config.experimental.get_memory_info(report["device"])
      report.update({
          "ending_current_bytes": int(observed["current"]),
          "peak_bytes": int(observed["peak"]),
          "peak_above_starting_current_bytes": max(
              0, int(observed["peak"]) - report["starting_current_bytes"]),
      })
    except (ValueError, RuntimeError) as error:
      report.update({"status": "unavailable", "reason": str(error)})
  return report


def measure(step, steps: int, warmup: int, batch_size: int, *,
            memory_device=None, iteration_output_index=None) -> dict:
  """Warms up, then times completed steps using a host-read barrier."""
  if steps < 1 or warmup < 1:
    raise ValueError("steps and warmup must be positive")

  def synchronized_step():
    outputs = [value.numpy() for value in tf.nest.flatten(step())]
    for value in outputs:
      if not np.isfinite(value).all():
        raise FloatingPointError("benchmark step produced non-finite outputs")
    return outputs

  for _ in range(warmup):
    outputs = synchronized_step()
  first_iteration = (int(outputs[iteration_output_index])
                     if iteration_output_index is not None else None)
  previous_iteration = first_iteration
  memory = _memory_start(memory_device)
  latencies = []
  for _ in range(steps):
    start = time.perf_counter()
    outputs = synchronized_step()
    if iteration_output_index is not None:
      iteration = int(outputs[iteration_output_index])
      if iteration != previous_iteration + 1:
        raise RuntimeError(
            "Measured training step skipped an optimizer update; increase "
            "warmup or investigate non-finite gradients/loss scaling")
      previous_iteration = iteration
    latencies.append(time.perf_counter() - start)
  result = {**summarize_latencies(latencies, batch_size),
            "memory": _memory_finish(memory)}
  if iteration_output_index is not None:
    result["optimizer_iterations"] = {
        "after_warmup": first_iteration,
        "after_measurement": previous_iteration,
        "measured_updates": previous_iteration - first_iteration,
    }
  return result


@contextlib.contextmanager
def strict_device(device):
  """Rejects missing requested devices and restores the caller's placement policy."""
  previous = tf.config.get_soft_device_placement()
  try:
    if device:
      specification = tf.DeviceSpec.from_string(device)
      kind, index = specification.device_type, specification.device_index or 0
      available = tf.config.list_logical_devices(kind)
      if kind not in ("CPU", "GPU") or not any(
          tf.DeviceSpec.from_string(item.name).device_index == index
          for item in available
      ):
        raise ValueError(f"Requested device is unavailable: {device}")
      tf.config.set_soft_device_placement(False)
    with tf.device(device) if device else contextlib.nullcontext():
      probe = None
      if device:
        product = tf.matmul(tf.ones((2, 2)), tf.ones((2, 2)))
        product.numpy()
        actual = tf.DeviceSpec.from_string(product.device)
        if actual.device_type != kind or actual.device_index != index:
          raise RuntimeError(f"Device probe ran on {product.device}, not {device}")
        probe = product.device
      yield probe
  finally:
    tf.config.set_soft_device_placement(previous)


def source_provenance():
  """Captures the implementation identity without reading credentials or data."""
  package = Path(__file__).resolve().parent
  root = package.parents[1]
  report = {"module_sha256": {
      name: hashlib.sha256((package / name).read_bytes()).hexdigest()
      for name in ("benchmark.py", "model.py", "layers.py", "train.py",
                   "losses.py", "data.py", "constants.py", "checkpoint.py")}}
  for key, arguments in (("commit", ["rev-parse", "HEAD"]),
                         ("worktree_status", ["status", "--porcelain"])):
    try:
      result = subprocess.run(["git", "-C", str(root), *arguments],
                              capture_output=True, text=True, timeout=5, check=False)
      report[key] = result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
      report[key] = None
  return report


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
    with strict_device(args.device) as probe_device:
      tf.keras.utils.set_random_seed(args.seed)
      inputs, labels = synthetic_batch(args.batch_size, args.tile_size)
      gpu_devices = tf.config.list_physical_devices("GPU")
      requested = tf.DeviceSpec.from_string(args.device) if args.device else None
      memory_device = None
      if requested is not None and requested.device_type == "GPU":
        memory_device = f"GPU:{requested.device_index or 0}"
      elif requested is None and tf.config.list_logical_devices("GPU"):
        memory_device = "GPU:0"
      weights = getattr(args, "weights", None)
      loaded_sha256 = None

      def build_model():
        nonlocal loaded_sha256
        model = MambaForestSegmenter(**config)
        tf.reduce_sum(model(inputs, training=False)).numpy()
        if weights is not None:
          digest = weights_sha256(weights)
          model.load_weights(weights)
          if weights_sha256(weights) != digest:
            raise ValueError("Checkpoint changed while loading benchmark weights")
          if loaded_sha256 is not None and loaded_sha256 != digest:
            raise ValueError("Checkpoint changed between benchmark models")
          loaded_sha256 = digest
        return model

      metadata = {
          "tensorflow_version": tf.__version__,
          "tensorflow_build": tf.sysconfig.get_build_info(),
          "python_version": platform.python_version(),
          "platform": platform.platform(),
          "cpu_count": os.cpu_count(),
          "source": source_provenance(),
          "gpu_details": [
              {"name": device.name,
               **tf.config.experimental.get_device_details(device)}
              for device in gpu_devices],
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
          "requested_device": args.device, "probe_device": probe_device,
          "soft_device_placement": tf.config.get_soft_device_placement(),
          "input_devices": {key: value.device for key, value in inputs.items()},
          "precision": args.precision, "seed": args.seed,
          "input_shapes": {key: value.shape.as_list() for key, value in inputs.items()},
          "label_shape": labels.shape.as_list(), "model_config": config,
          "warmup_steps": args.warmup, "steps": args.steps,
          "weights": str(weights) if weights is not None else None,
          "memory_scope": (
              "Per-mode TensorFlow allocator peak reset after warmup, including "
              "live models, inputs and optimizer slots. Both mode retains the "
              "inference model during training; use separate processes/modes "
              "for isolated memory comparisons. Not total/reserved GPU VRAM."),
          "scope": ("Synthetic model compute plus scalar output synchronization; "
                    "excludes dataset IO, preprocessing, checkpoint loading, "
                    "graph tracing and output-file writes. No accuracy measurement."),
      }

      if args.mode in ("inference", "both"):
        model = build_model()
        metadata["parameters"] = model.count_params()

        def infer():
          # Scalar reduction forces completion without copying a probability map.
          return tf.reduce_sum(model(inputs, training=False))

        modes = ("eager", "graph") if args.inference_mode == "both" else (
            args.inference_mode,)
        for mode in modes:
          step = infer if mode == "eager" else tf.function(infer)
          results[f"inference_{mode}"] = measure(
              step, args.steps, args.warmup, args.batch_size,
              memory_device=memory_device)

      if args.mode in ("training", "both"):
        # Training-only mode creates exactly one model. Both mode keeps a
        # separate inference model so training cannot change its measurements.
        tf.keras.utils.set_random_seed(args.seed)
        train_model = build_model()
        metadata["parameters"] = train_model.count_params()
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
            train, args.steps, args.warmup, args.batch_size,
            memory_device=memory_device, iteration_output_index=-1)
        metadata["training_optimizer"] = {
            "name": "AdamW", "learning_rate": 2e-4,
            "weight_decay": 5e-3, "global_clipnorm": 1.0,
            "dynamic_loss_scaling": args.precision == "mixed_float16",
        }
        metadata["training_loss"] = loss_fn.get_config()
      metadata["weights_sha256"] = loaded_sha256
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
