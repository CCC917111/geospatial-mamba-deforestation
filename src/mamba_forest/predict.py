#!/usr/bin/env python3
"""Run a trained CNN-Mamba segmenter on your own satellite tiles.

This is the entry point for applying the model to new data rather than to the
ForTy v1 benchmark splits: point it at a directory of ``.npz`` tiles, and it
writes a forest-type map for each tile plus a table of how much of each class
it found.

Input format
------------
One ``.npz`` file per tile, with three arrays:

    s2         (T, H, W, 10)  Sentinel-2 seasonal composites, raw reflectance
    elevation  (H, W, 3)      elevation in metres, slope and aspect in degrees
    climate    (T, 14)        the 14 climate variables, one row per composite

``T`` must be 4 (the four seasonal composites the model was trained with), and
``H`` and ``W`` must be equal and divisible by 8; 128 x 128 is what the model
saw during training. The arrays are stored unnormalised: this script applies
exactly the same normalisation as the training pipeline, so a tile prepared for
training and a tile prepared for prediction go through the same code path.

Example:
    python -m mamba_forest.predict \
        --weights runs/mamba_forest/best.weights.h5 \
        --input-dir my_tiles --output-dir predictions
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import tensorflow as tf

from mamba_forest import checkpoint
from mamba_forest import data as data_lib
from mamba_forest.model import MambaForestSegmenter
from mamba_forest.constants import DEFAULT_MODEL_VARIANT

REQUIRED_ARRAYS = ("s2", "elevation", "climate")

# One colour per class, in the label order of data_lib.CLASS_NAMES.
CLASS_COLORS = (
    (200, 200, 200),  # Unknown
    (27, 94, 32),     # Natural Forest
    (102, 187, 106),  # Planted Forest
    (174, 213, 129),  # Tree Crops
    (220, 231, 117),  # Other Vegetation
    (66, 165, 245),   # Water
    (224, 247, 250),  # Ice
    (188, 170, 134),  # Bare Ground
    (239, 83, 80),    # Built Areas
)


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--weights", required=True, help="Path to a .weights.h5 file")
  p.add_argument("--input-dir", required=True,
                 help="Directory of .npz tiles (see the format above)")
  p.add_argument("--output-dir", required=True,
                 help="Where the class maps and the summary table are written")
  p.add_argument("--pattern", default="*.npz",
                 help="Glob used to select tiles inside --input-dir")
  p.add_argument("--save-png", action="store_true",
                 help="Also write a colour-mapped PNG next to each class map")
  p.add_argument("--save-probabilities", action="store_true",
                 help="Also write the full per-class probability tensor per tile")
  p.add_argument("--timings", type=Path,
                 help="Write stage timings to a new JSON file")
  checkpoint.add_model_arguments(p)
  args = p.parse_args()
  try:
    checkpoint.resolve_model_arguments(args)
  except (ValueError, OSError) as error:
    p.error(str(error))
  return args


def load_tile(path: Path) -> dict[str, np.ndarray]:
  """Reads one ``.npz`` tile and checks it against the documented contract."""
  with np.load(path) as handle:
    missing = [key for key in REQUIRED_ARRAYS if key not in handle]
    if missing:
      raise ValueError(f"{path.name}: missing array(s) {missing}; "
                       f"expected {list(REQUIRED_ARRAYS)}")
    tile = {key: np.asarray(handle[key], dtype=np.float32)
            for key in REQUIRED_ARRAYS}

  if tile["s2"].ndim != 4 or tile["s2"].shape[-1] != data_lib.FEATURE_SHAPES["s2"][-1]:
    raise ValueError(f"{path.name}: s2 must be (T, H, W, "
                     f"{data_lib.FEATURE_SHAPES['s2'][-1]}), got {tile['s2'].shape}")
  time, height, width, _ = tile["s2"].shape
  if time != data_lib.FEATURE_SHAPES["s2"][0]:
    raise ValueError(f"{path.name}: expected four seasonal composites, got {time}")
  if height < 1 or width < 1:
    raise ValueError(f"{path.name}: tile dimensions must be positive")

  if tile["elevation"].shape != (height, width, 3):
    raise ValueError(f"{path.name}: elevation must be ({height}, {width}, 3), "
                     f"got {tile['elevation'].shape}")
  if tile["climate"].shape != (time, data_lib.FEATURE_SHAPES["climate"][-1]):
    raise ValueError(f"{path.name}: climate must be ({time}, "
                     f"{data_lib.FEATURE_SHAPES['climate'][-1]}), "
                     f"got {tile['climate'].shape}")
  if height != width:
    raise ValueError(f"{path.name}: tiles must be square, got {height} x {width}")
  if height % 8:
    raise ValueError(f"{path.name}: the tile size must be divisible by 8 "
                     f"(the encoder pools three times), got {height}")
  for key, array in tile.items():
    if not np.isfinite(array).all():
      raise ValueError(f"{path.name}: {key} contains non-finite values")
  return tile


def preprocess_tile(tile: dict[str, np.ndarray]) -> dict[str, tf.Tensor]:
  """Applies the training-time normalisation and adds the batch axis."""
  example = {key: tf.constant(value) for key, value in tile.items()}
  normalised = data_lib.normalize(example)
  return {key: tf.expand_dims(normalised[key], axis=0)
          for key in ("s2", "elevation", "climate")}


def make_predictor(model, save_probabilities: bool = False):
  """Compiles inference for any valid tile size without tracing each size.

  Returns maps [B,H,W] and mean confidence [B]. The full probability tensor
  [B,H,W,C] is returned only when requested, keeping the usual host transfer
  small. The model must already be built before creating this function.
  """
  time, _, _, channels = data_lib.FEATURE_SHAPES["s2"]
  signature = {
      "s2": tf.TensorSpec([None, time, None, None, channels], tf.float32),
      "elevation": tf.TensorSpec(
          [None, None, None, data_lib.FEATURE_SHAPES["elevation"][-1]], tf.float32),
      "climate": tf.TensorSpec(
          [None, *data_lib.FEATURE_SHAPES["climate"]], tf.float32),
  }

  @tf.function(input_signature=[signature], reduce_retracing=True)
  def predict(inputs):
    probabilities = model(inputs, training=False)
    outputs = {
        "class_map": tf.cast(tf.argmax(probabilities, axis=-1), tf.uint8),
        "mean_confidence": tf.reduce_mean(
            tf.reduce_max(probabilities, axis=-1), axis=[1, 2]),
    }
    if save_probabilities:
      outputs["probabilities"] = probabilities
    return outputs

  return predict


def colourise(class_map: np.ndarray) -> np.ndarray:
  """Maps a [H, W] array of class indices to an [H, W, 3] uint8 image."""
  palette = np.array(CLASS_COLORS, dtype=np.uint8)
  return palette[np.clip(class_map, 0, len(palette) - 1)]


def class_shares(class_map: np.ndarray, num_classes: int) -> np.ndarray:
  """Fraction of the tile covered by each class."""
  counts = np.bincount(class_map.reshape(-1), minlength=num_classes)
  return counts / max(class_map.size, 1)


def _read_inputs(path):
  started = time.perf_counter()
  tile = load_tile(path)
  source = {"path": str(path), "size_bytes": path.stat().st_size,
            "shapes": {key: list(value.shape) for key, value in tile.items()}}
  read_seconds = time.perf_counter() - started
  started = time.perf_counter()
  inputs = preprocess_tile(tile)
  return inputs, source, read_seconds, time.perf_counter() - started


def _runtime_metadata(args, sources):
  return {
      "weights": str(args.weights),
      "weights_sha256": checkpoint.weights_sha256(args.weights),
      "training_config_path": getattr(args, "training_config_path", None),
      "output_dir": str(args.output_dir), "input_dir": str(args.input_dir),
      "pattern": args.pattern, "source_file_count": len(sources),
      "sources": list(sources.values()),
      "save_options": {"png": args.save_png, "probabilities": args.save_probabilities},
      "model_config": {name: getattr(args, name) for name in (
          "d_model", "dropout", "d_state_temporal", "d_state_spatial",
          "bottleneck_channels")} | {
              "model_variant": getattr(args, "model_variant", DEFAULT_MODEL_VARIANT)},
      "tensorflow_version": tf.__version__, "python_version": platform.python_version(),
      "platform": platform.platform(), "cpu_count": os.cpu_count(),
      "devices": [{"name": device.name, "type": device.device_type}
                  for device in tf.config.list_physical_devices()],
      "precision": tf.keras.mixed_precision.global_policy().name,
      "thread_settings": {name: os.environ.get(name) for name in (
          "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS")},
  }


def run_prediction(args, *, repeats=1, warmup=0, quiet=False):
  """Runs the production tile path and returns raw, synchronized stage timings.

  ``warmup`` and ``repeats`` count complete passes over the input files. Warmup
  passes also export outputs but are excluded from measured samples. Multiple
  passes use separate subdirectories; the normal one-pass CLI keeps its flat
  output layout. Arrays are held for one tile at a time. Timing starts inside
  this function, after imports, argument parsing and configuration resolution.
  """
  if repeats < 1 or warmup < 0:
    raise ValueError("repeats must be positive and warmup must be non-negative")
  timings_path = getattr(args, "timings", None)
  if timings_path is not None:
    timings_path = Path(timings_path)
    if timings_path.suffix != ".json" or timings_path.exists():
      raise ValueError("--timings must name a new .json file")
  total_started = time.perf_counter()
  num_classes = data_lib.NUM_CLASSES
  input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  paths = sorted(input_dir.glob(args.pattern))
  if not paths:
    raise ValueError(f"no tiles matching {args.pattern!r} in {input_dir}")
  if len({path.stem for path in paths}) != len(paths):
    raise ValueError("selected input tiles must have unique stems")
  if not quiet:
    print(f"found {len(paths)} tile(s) in {input_dir}")
  first_pass_started = time.perf_counter()
  prepared = _read_inputs(paths[0])
  first_inputs = prepared[0]
  started = time.perf_counter()
  model = MambaForestSegmenter(
      num_classes=num_classes, d_model=args.d_model, dropout_rate=args.dropout,
      d_state_temporal=args.d_state_temporal,
      d_state_spatial=args.d_state_spatial,
      bottleneck_channels=args.bottleneck_channels,
      model_variant=getattr(args, "model_variant", DEFAULT_MODEL_VARIANT))
  # Synchronize the eager build before stopping its timer or loading weights.
  tf.reduce_sum(model(first_inputs, training=False)).numpy()
  model.load_weights(args.weights)
  model_load_seconds = time.perf_counter() - started
  del first_inputs
  predict = make_predictor(model, args.save_probabilities)
  if not quiet:
    print(f"loaded weights from {args.weights}")
  started = time.perf_counter()
  if args.save_png:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import image as mpimg
  export_setup_seconds = time.perf_counter() - started
  sources, samples, passes = {}, [], []
  cold_start_seconds = None
  for pass_index in range(warmup + repeats):
    phase = "warmup" if pass_index < warmup else "measured"
    index = pass_index if phase == "warmup" else pass_index - warmup
    pass_started = first_pass_started if pass_index == 0 else time.perf_counter()
    directory = (output_dir if warmup + repeats == 1 else
                 output_dir / f"{phase}_{index:03d}")
    directory.mkdir(parents=True, exist_ok=warmup + repeats == 1)
    totals = np.zeros(num_classes, dtype=np.float64)
    table_path = directory / "predictions.csv"
    csv_seconds = 0.0
    opened = time.perf_counter()
    with table_path.open("w", newline="", encoding="utf-8") as handle:
      csv_seconds += time.perf_counter() - opened
      writer = None
      for tile_index, path in enumerate(paths):
        if prepared is None:
          prepared = _read_inputs(path)
        inputs, source, read_seconds, preprocess_seconds = prepared
        prepared = None
        sources[str(path)] = source
        started = time.perf_counter()
        # Materialize exactly the tensors used by the production exporter.
        arrays = {key: value.numpy() for key, value in predict(inputs).items()}
        inference_seconds = time.perf_counter() - started
        if not samples:
          cold_start_seconds = time.perf_counter() - total_started
        started = time.perf_counter()
        class_map = arrays["class_map"][0]
        shares = class_shares(class_map, num_classes)
        totals += shares
        forest = float(shares[list(data_lib.FOREST_CLASS_INDICES)].sum())
        row = {
            "tile": path.stem, "height": class_map.shape[0],
            "width": class_map.shape[1],
            "mean_confidence": float(arrays["mean_confidence"][0]),
            "forest_share": forest,
            **{f"share_{name.lower().replace(' ', '_')}": float(value)
               for name, value in zip(data_lib.CLASS_NAMES, shares)},
        }
        png = colourise(class_map) if args.save_png else None
        postprocess_seconds = time.perf_counter() - started
        started = time.perf_counter()
        np.save(directory / f"{path.stem}_class_map.npy", class_map)
        if args.save_probabilities:
          np.save(directory / f"{path.stem}_probabilities.npy",
                  arrays["probabilities"][0].astype(np.float32, copy=False))
        if args.save_png:
          mpimg.imsave(directory / f"{path.stem}_class_map.png", png)
        export_seconds = time.perf_counter() - started
        started = time.perf_counter()
        if writer is None:
          writer = csv.DictWriter(handle, fieldnames=list(row))
          writer.writeheader()
        writer.writerow(row)
        csv_seconds += time.perf_counter() - started
        sample = {
            "phase": phase, "pass_index": index, "tile_index": tile_index,
            "path": str(path), "output_dir": str(directory),
            "first_inference": not samples,
            "read_validate_seconds": read_seconds,
            "preprocess_seconds": preprocess_seconds,
            "inference_sync_seconds": inference_seconds,
            "postprocess_seconds": postprocess_seconds,
            "array_image_export_seconds": export_seconds,
        }
        sample["pipeline_seconds"] = sum(sample[key] for key in (
            "read_validate_seconds", "preprocess_seconds", "inference_sync_seconds",
            "postprocess_seconds", "array_image_export_seconds"))
        samples.append(sample)
        if not quiet:
          print(f"  {path.stem}: forest {100 * forest:5.1f}% | "
                f"natural {100 * shares[data_lib.NATURAL_FOREST_INDEX]:4.1f}% | "
                f"planted {100 * shares[data_lib.PLANTED_FOREST_INDEX]:4.1f}% | "
                f"tree crops {100 * shares[data_lib.TREE_CROPS_INDEX]:4.1f}%")
        del inputs, arrays
      started = time.perf_counter()
    csv_seconds += time.perf_counter() - started  # Includes buffered close/flush.
    passes.append({"phase": phase, "pass_index": index, "tiles": len(paths),
                   "csv_export_seconds": csv_seconds,
                   "wall_seconds": time.perf_counter() - pass_started,
                   "includes_initialization": pass_index == 0})
    if not quiet:
      print("\naverage composition over all tiles:")
      for name, value in zip(data_lib.CLASS_NAMES, totals / len(paths)):
        print(f"  {name:18s} {100 * value:5.2f}%")
      print(f"\nclass maps and {table_path.name} written to {directory}")
  report = {
      "schema_version": 1, "metadata": _runtime_metadata(args, sources),
      "scope": {
          "start": ("run_prediction entry; excludes imports, CLI parsing "
                    "and config resolution"),
          "synchronization": "all actual predictor return tensors copied to NumPy",
          "first_inference": ("includes first graph tracing and execution; "
                              "never steady state"),
          "writes": ("file writes and close/flush to OS cache; "
                     "no fsync or physical-disk claim"),
          "warmup": ("complete input passes including exports; recorded "
                     "but excluded from measurements"),
          "repeats": "complete measured input passes with one model and predictor",
          "pipeline_seconds": ("sum of tile read/validation, preprocessing, "
                               "inference/sync, postprocessing and array/image export; "
                               "excludes CSV, directory operations and initialization"),
          "cache": ("OS file caches are not cleared; "
                    "repeated source reads may be cached"),
          "report_write": "timing-report JSON serialization/write is excluded",
          "accuracy": "no trained accuracy or real-data/GPU performance is established",
      },
      "repeats": repeats, "warmup_passes": warmup,
      "model_build_load_seconds": model_load_seconds,
      "cold_start_to_first_prediction_seconds": cold_start_seconds,
      "export_setup_seconds": export_setup_seconds,
      "samples": samples, "passes": passes,
      "first_inference_sync_seconds": samples[0]["inference_sync_seconds"],
  }
  report["total_wall_seconds"] = time.perf_counter() - total_started
  if timings_path is not None:
    timings_path.parent.mkdir(parents=True, exist_ok=True)
    with timings_path.open("x", encoding="utf-8") as handle:
      json.dump(report, handle, indent=2, allow_nan=False)
      handle.write("\n")
  return report


def main() -> None:
  run_prediction(parse_args())


if __name__ == "__main__":
  main()
