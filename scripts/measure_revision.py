"""Measure one source revision with shared synthetic inputs and checkpoint.

Execute this same script with PYTHONPATH set to the revision's src directory.
Each measurement is a fresh process containing one model and one workload.
No dataset or semantic class mapping is needed. Timings measure synthetic
model compute, not real-data accuracy or file-to-file inference.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time

import numpy as np


def digest(path):
  value = hashlib.sha256()
  with Path(path).open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
      value.update(block)
  return value.hexdigest()


def write_json(path, payload):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(path.name + ".tmp")
  temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
  temporary.replace(path)


def source_metadata():
  import mamba_forest

  directory = Path(mamba_forest.__file__).resolve().parent
  root = directory.parents[1]
  result = {"package_directory": str(directory), "module_sha256": {
      name: digest(directory / name) for name in (
          "model.py", "layers.py", "train.py", "losses.py", "constants.py")}}
  for name, command in (("commit", ["rev-parse", "HEAD"]),
                        ("worktree_status", ["status", "--short"])):
    completed = subprocess.run(["git", "-C", str(root), *command],
                               capture_output=True, text=True, check=False)
    result[name] = completed.stdout.strip() if not completed.returncode else None
  return result


def configure_device(tf, device):
  tf.config.set_soft_device_placement(False)
  gpus = tf.config.list_physical_devices("GPU")
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  if device.startswith("/GPU:"):
    index = int(device.rsplit(":", 1)[1])
    if index >= len(gpus):
      raise RuntimeError(f"Requested {device}, but only {len(gpus)} GPUs are visible")
  with tf.device(device):
    product = tf.matmul(tf.ones((2, 2)), tf.ones((2, 2)))
  product.numpy()
  if not product.device.endswith(device.removeprefix("/")):
    raise RuntimeError(f"Device probe ran on {product.device}, expected {device}")
  return {"tensorflow_version": tf.__version__,
          "python_version": platform.python_version(),
          "requested_device": device, "probe_device": product.device,
          "gpu_details": [tf.config.experimental.get_device_details(gpu)
                          for gpu in gpus],
          "thread_environment": {name: os.environ.get(name) for name in (
              "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS")}}


def make_model(tf, config, inputs, weights=None):
  from mamba_forest.model import MambaForestSegmenter

  model = MambaForestSegmenter(**config)
  model(inputs, training=False).numpy()
  if weights is not None:
    model.load_weights(weights)
  return model


def prepare(args, tf, runtime):
  directory = args.fixture_dir.resolve()
  directory.mkdir(parents=True, exist_ok=False)
  rng = np.random.default_rng(args.seed)
  size, batch = args.tile_size, args.batch_size
  arrays = {
      "s2": rng.uniform(-1, 1, (batch, 4, size, size, 10)).astype(np.float32),
      "elevation": rng.normal(size=(batch, size, size, 3)).astype(np.float32),
      "climate": rng.normal(size=(batch, 4, 14)).astype(np.float32),
      "labels": rng.integers(0, 9, (batch, size, size), dtype=np.int32),
  }
  np.savez_compressed(directory / "inputs.npz", **arrays)
  config = {"num_classes": 9, "d_model": args.d_model,
            "dropout_rate": args.dropout, "d_state_temporal": args.d_state_temporal,
            "d_state_spatial": args.d_state_spatial,
            "bottleneck_channels": args.bottleneck_channels,
            "model_variant": args.model_variant}
  inputs = {key: tf.constant(value) for key, value in arrays.items() if key != "labels"}
  weights = directory / "initial.weights.h5"
  tf.keras.utils.set_random_seed(args.seed)
  tf.keras.mixed_precision.set_global_policy("float32")
  model = make_model(tf, config, inputs, args.weights)
  model.save_weights(weights)
  del model
  gc.collect()
  golden = {}
  for precision in args.precisions:
    tf.keras.mixed_precision.set_global_policy(precision)
    model = make_model(tf, config, inputs, weights)
    path = directory / ("golden-" + precision + ".npy")
    np.save(path, model(inputs, training=False).numpy())
    golden[precision] = {"file": path.name, "sha256": digest(path)}
    del model
    gc.collect()
  result = {"schema_version": 1,
            "scope": "Shared synthetic compute fixture; no accuracy claim",
            "runtime": runtime, "source": source_metadata(), "seed": args.seed,
            "model_config": config,
            "input_shapes": {key: list(value.shape) for key, value in arrays.items()},
            "inputs_sha256": digest(directory / "inputs.npz"),
            "weights_sha256": digest(weights), "golden": golden,
            "supplied_weights": str(args.weights) if args.weights else None}
  write_json(directory / "fixture.json", result)
  return result


def memory_info(tf, device):
  if not device.startswith("/GPU:"):
    return None
  return tf.config.experimental.get_memory_info(device.removeprefix("/"))


def capture_profile(tf, directory, steps, synchronized_step, consume, mode):
  """Capture warmed steps without including profiler overhead in timings."""
  directory = Path(directory).resolve()
  directory.mkdir(parents=True, exist_ok=False)
  options = tf.profiler.experimental.ProfilerOptions(
      host_tracer_level=2, python_tracer_level=0, device_tracer_level=1)
  started = False
  try:
    tf.profiler.experimental.start(str(directory), options=options)
    started = True
    for index in range(steps):
      trace = tf.profiler.experimental.Trace(mode + "_step", step_num=index, _r=1)
      with trace:
        consume(synchronized_step())
  finally:
    if started:
      tf.profiler.experimental.stop()
  artifacts = []
  for path in sorted(directory.rglob("*")):
    if path.is_file() and not path.is_symlink():
      artifacts.append({"path": str(path.relative_to(directory)),
                        "bytes": path.stat().st_size, "sha256": digest(path)})
  if not any(item["path"].endswith(".xplane.pb") and item["bytes"] > 0
             for item in artifacts):
    raise RuntimeError("Profiler produced no nonempty .xplane.pb artifact")
  return {"status": "captured", "directory": str(directory),
          "steps": steps, "artifacts": artifacts,
          "tracer_options": {"host_tracer_level": 2, "python_tracer_level": 0,
                             "device_tracer_level": 1}}


def layer_scopes(model):
  """Record existing names for trace navigation without renaming any layer."""
  names = ("temporal_mamba", "temporal_norm", "optical_proj", "elevation_proj",
           "climate_proj", "fusion_layer", "enc1", "enc2", "enc3",
           "bottleneck_conv", "spatial_mamba", "spatial_norm", "sasf",
           "up1", "dec1", "up2", "dec2", "up3", "dec3", "head_conv",
           "head_output")
  return {name: {"name": layer.name, "path": getattr(layer, "path", None),
                 "class": type(layer).__name__}
          for name in names if (layer := getattr(model, name, None)) is not None}


def measure(args, tf, runtime):
  from mamba_forest.losses import ComboLoss
  from mamba_forest.train import scale_loss, unscale_gradients

  directory = args.fixture_dir.resolve()
  fixture = json.loads((directory / "fixture.json").read_text())
  weights = directory / "initial.weights.h5"
  if digest(weights) != fixture["weights_sha256"]:
    raise ValueError("Shared weights changed")
  if digest(directory / "inputs.npz") != fixture["inputs_sha256"]:
    raise ValueError("Inputs changed")
  tf.keras.utils.set_random_seed(fixture["seed"])
  tf.keras.mixed_precision.set_global_policy(args.precision)
  with np.load(directory / "inputs.npz", allow_pickle=False) as stored:
    arrays = {key: stored[key] for key in ("s2", "elevation", "climate", "labels")}
  stored_shapes = {key: list(value.shape) for key, value in arrays.items()}
  if stored_shapes != fixture["input_shapes"]:
    raise ValueError("Stored input shapes differ from fixture metadata")
  fixture_batch = arrays["s2"].shape[0]
  if any(value.shape[0] != fixture_batch for value in arrays.values()):
    raise ValueError("Fixture inputs do not share one batch dimension")
  if args.repeat_batch is not None:
    if fixture_batch != 1:
      raise ValueError("--repeat-batch requires a one-tile fixture")
    arrays = {key: np.repeat(value, args.repeat_batch, axis=0)
              for key, value in arrays.items()}
  batch = arrays["s2"].shape[0]
  input_shapes = {key: list(value.shape) for key, value in arrays.items()}
  inputs = {key: tf.constant(arrays[key]) for key in ("s2", "elevation", "climate")}
  labels = tf.one_hot(arrays["labels"], 9)
  del arrays
  model = make_model(tf, fixture["model_config"], inputs, weights)
  observed = model(inputs, training=False).numpy()
  golden_record = fixture["golden"][args.precision]
  golden_path = directory / golden_record["file"]
  if digest(golden_path) != golden_record["sha256"]:
    raise ValueError("Golden predictions changed")
  expected = np.load(golden_path, allow_pickle=False)
  if args.repeat_batch is not None:
    expected = np.repeat(expected, args.repeat_batch, axis=0)
  tolerance = ((2e-5, 2e-6) if args.precision == "float32" else (5e-3, 5e-4))
  np.testing.assert_allclose(observed, expected, rtol=tolerance[0], atol=tolerance[1])
  agreement = {"passed": True, "rtol": tolerance[0], "atol": tolerance[1],
               "max_absolute_error": float(np.max(np.abs(observed - expected))),
               "argmax_agreement": float(np.mean(
                   observed.argmax(-1) == expected.argmax(-1)))}
  del observed, expected
  if args.mode == "inference":
    @tf.function
    def step():
      return tf.reduce_sum(model(inputs, training=False))
    optimizer = None
  else:
    loss_fn = ComboLoss(9)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=2e-4, weight_decay=5e-3, global_clipnorm=1.0)
    if args.precision == "mixed_float16":
      optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    @tf.function
    def step():
      with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(labels, predictions)
        scaled = scale_loss(optimizer, loss)
      gradients = unscale_gradients(
          optimizer, tape.gradient(scaled, model.trainable_variables))
      optimizer.apply_gradients(
          (g, v) for g, v in zip(gradients, model.trainable_variables) if g is not None)
      return loss, tf.identity(optimizer.iterations)

  def synchronized_step():
    values = [value.numpy() for value in tf.nest.flatten(step())]
    if not all(np.isfinite(value).all() for value in values):
      raise FloatingPointError("Non-finite benchmark output")
    return values

  for _ in range(args.warmup):
    synchronized_step()
  last_iteration = int(optimizer.iterations.numpy()) if optimizer else None
  iterations_start = last_iteration
  losses = []

  def consume(values):
    nonlocal last_iteration
    if optimizer is not None:
      iteration = int(values[1])
      if iteration != last_iteration + 1:
        raise RuntimeError(
            "Optimizer skipped a measured update; increase warmup for both revisions")
      last_iteration = iteration
      losses.append(float(values[0]))

  metadata = {"schema_version": 1, "status": "passed", "phase": args.phase,
              "runtime": runtime, "source": source_metadata(),
              "mode": args.mode, "precision": args.precision,
              "model_config": fixture["model_config"],
              "parameters": model.count_params(),
              "weights_sha256": fixture["weights_sha256"],
              "inputs_sha256": fixture["inputs_sha256"],
              "forward_agreement": agreement, "batch_size": batch,
              "input_shapes": input_shapes,
              "batch_replication": {
                  "kind": ("repeat_single_fixture_tile"
                           if args.repeat_batch is not None else "none"),
                  "repeat_count": args.repeat_batch or 1,
                  "fixture_batch_size": fixture_batch},
              "warmup_steps": args.warmup,
              "warmup_optimizer_updates": iterations_start}
  if args.phase == "profile":
    profile = capture_profile(tf, args.profile_dir, args.profile_steps,
                              synchronized_step, consume, args.mode)
    profile.update({"layer_scopes": layer_scopes(model),
                    "gpu_trace_coverage": (
                        "unverified" if args.device.startswith("/GPU:")
                        else "not_applicable_on_cpu"),
                    "coverage_note": (
                        "A saved XPlane file does not prove GPU kernel coverage; "
                        "inspect device planes in the profiler viewer.")})
    if digest(weights) != fixture["weights_sha256"]:
      raise ValueError("Shared weights changed")
    return {**metadata, "steps": args.profile_steps, "profile": profile,
            "optimizer_updates": (last_iteration - iterations_start
                                  if optimizer else None),
            "training_loss_samples": losses,
            "scope": (
                "Warmed synthetic compute trace in a separate invocation. "
                "Profiler overhead is not a latency or throughput measurement. "
                "No accuracy, semantic class or total GPU VRAM claim.")}
  before = memory_info(tf, args.device)
  if before is not None:
    tf.config.experimental.reset_memory_stats(args.device.removeprefix("/"))
  latencies = []
  for _ in range(args.steps):
    start = time.perf_counter()
    values = synchronized_step()
    elapsed = time.perf_counter() - start
    consume(values)
    latencies.append(elapsed)
  after = memory_info(tf, args.device)
  memory = ({"status": "available", "starting_current_bytes": before["current"],
             "ending_current_bytes": after["current"], "peak_bytes": after["peak"],
             "peak_above_starting_current_bytes": after["peak"] - before["current"]}
            if before is not None else {"status": "not_applicable_on_cpu"})
  if digest(weights) != fixture["weights_sha256"]:
    raise ValueError("Shared weights changed")
  return {**metadata, "steps": args.steps,
          "latencies_seconds": latencies,
          "median_seconds": float(np.median(latencies)),
          "p95_seconds": float(np.percentile(latencies, 95)),
          "tiles_per_second": batch * args.steps / sum(latencies),
          "optimizer_updates": (last_iteration - iterations_start
                                if optimizer else None),
          "training_loss_samples": losses, "memory": memory,
          "scope": (
              "Synthetic compute and scalar synchronization, one model per process; "
              "excludes imports, tracing, data IO, preprocessing, loading and exports. "
              "Memory is TensorFlow allocator usage, not total GPU VRAM. "
              "Dropout draws may differ between equivalent output selections. "
              "No accuracy or semantic class claim.")}


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--phase", choices=("prepare", "measure", "profile"),
                      required=True)
  parser.add_argument("--fixture-dir", type=Path, required=True)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--device", default="/CPU:0")
  parser.add_argument("--mode", choices=("inference", "training"), default="inference")
  parser.add_argument("--precision", choices=("float32", "mixed_float16"),
                      default="float32")
  parser.add_argument("--precisions", nargs="+", choices=("float32", "mixed_float16"),
                      default=["float32"])
  parser.add_argument("--steps", type=int, default=30)
  parser.add_argument("--warmup", type=int, default=5)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--weights", type=Path)
  parser.add_argument("--tile-size", type=int, default=128)
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--repeat-batch", type=int)
  parser.add_argument("--profile-dir", type=Path)
  parser.add_argument("--profile-steps", type=int, default=3)
  parser.add_argument("--d-model", type=int, default=32)
  parser.add_argument("--d-state-temporal", type=int, default=8)
  parser.add_argument("--d-state-spatial", type=int, default=16)
  parser.add_argument("--bottleneck-channels", type=int, default=512)
  parser.add_argument("--dropout", type=float, default=0.2)
  parser.add_argument("--model-variant", choices=("cnn_ssm", "cnn_only"),
                      default="cnn_ssm")
  args = parser.parse_args(argv)
  if args.phase in ("measure", "profile") and args.output is None:
    parser.error("--output is required for measurement or profiling")
  if args.phase == "profile" and args.profile_dir is None:
    parser.error("--profile-dir is required for profiling")
  if args.phase != "profile" and args.profile_dir is not None:
    parser.error("--profile-dir is only valid for profiling")
  if args.phase == "prepare" and args.repeat_batch is not None:
    parser.error("--repeat-batch is only valid for measurement or profiling")
  if args.repeat_batch is not None and args.repeat_batch < 1:
    parser.error("--repeat-batch must be positive")
  if min(args.steps, args.warmup, args.tile_size, args.batch_size, args.d_model,
         args.d_state_temporal, args.d_state_spatial, args.bottleneck_channels,
         args.profile_steps) < 1:
    parser.error("budgets and dimensions must be positive")
  if args.tile_size % 8 or not np.isfinite(args.dropout) or not 0 <= args.dropout < 1:
    parser.error("tile size must be divisible by 8; dropout must be finite in [0,1)")
  if re.fullmatch(r"/(?:GPU|CPU):[0-9]+", args.device) is None:
    parser.error("--device must be an explicit /CPU:N or /GPU:N")
  return args


def main():
  args = parse_args()
  if args.output is not None and args.output.exists():
    raise FileExistsError(f"Refusing to overwrite {args.output}")
  if args.profile_dir is not None and args.profile_dir.exists():
    raise FileExistsError(f"Refusing to overwrite {args.profile_dir}")
  import tensorflow as tf

  result = {"status": "failed", "phase": args.phase}
  try:
    runtime = configure_device(tf, args.device)
    with tf.device(args.device):
      result = (prepare(args, tf, runtime) if args.phase == "prepare"
                else measure(args, tf, runtime))
  except Exception as error:
    result["error_type"] = type(error).__name__
    result["error"] = f"{type(error).__name__}: {error}"
    raise
  finally:
    if args.output is not None:
      write_json(args.output, result)
  print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
  main()
