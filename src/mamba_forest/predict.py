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
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from mamba_forest import data as data_lib
from mamba_forest.model import MambaForestSegmenter

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
  p.add_argument("--d-model", type=int, default=32,
                 help="Must match the value used for training")
  p.add_argument("--dropout", type=float, default=0.2)
  p.add_argument("--d-state-temporal", type=int, default=8,
                 help="Must match the value used for training")
  p.add_argument("--d-state-spatial", type=int, default=16,
                 help="Must match the value used for training")
  p.add_argument("--bottleneck-channels", type=int, default=512,
                 help="Must match the value used for training")
  return p.parse_args()


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
  return tile


def preprocess_tile(tile: dict[str, np.ndarray]) -> dict[str, tf.Tensor]:
  """Applies the training-time normalisation and adds the batch axis."""
  example = {key: tf.constant(value) for key, value in tile.items()}
  normalised = data_lib.normalize(example)
  return {key: tf.expand_dims(normalised[key], axis=0)
          for key in ("s2", "elevation", "climate")}


def colourise(class_map: np.ndarray) -> np.ndarray:
  """Maps a [H, W] array of class indices to an [H, W, 3] uint8 image."""
  palette = np.array(CLASS_COLORS, dtype=np.uint8)
  return palette[np.clip(class_map, 0, len(palette) - 1)]


def class_shares(class_map: np.ndarray, num_classes: int) -> np.ndarray:
  """Fraction of the tile covered by each class."""
  counts = np.bincount(class_map.reshape(-1), minlength=num_classes)
  return counts / max(class_map.size, 1)


def main() -> None:
  args = parse_args()
  num_classes = data_lib.NUM_CLASSES
  input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  paths = sorted(input_dir.glob(args.pattern))
  if not paths:
    raise SystemExit(f"no tiles matching {args.pattern!r} in {input_dir}")
  print(f"found {len(paths)} tile(s) in {input_dir}")

  model = MambaForestSegmenter(
      num_classes=num_classes, d_model=args.d_model, dropout_rate=args.dropout,
      d_state_temporal=args.d_state_temporal,
      d_state_spatial=args.d_state_spatial,
      bottleneck_channels=args.bottleneck_channels)
  model(preprocess_tile(load_tile(paths[0])), training=False)  # build the weights
  model.load_weights(args.weights)
  print(f"loaded weights from {args.weights}")

  if args.save_png:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import image as mpimg

  rows = []
  totals = np.zeros(num_classes, dtype=np.float64)
  for path in paths:
    inputs = preprocess_tile(load_tile(path))
    probabilities = model(inputs, training=False).numpy()[0]
    class_map = probabilities.argmax(axis=-1).astype(np.uint8)
    confidence = probabilities.max(axis=-1)

    stem = path.stem
    np.save(output_dir / f"{stem}_class_map.npy", class_map)
    if args.save_probabilities:
      np.save(output_dir / f"{stem}_probabilities.npy",
              probabilities.astype(np.float32))
    if args.save_png:
      mpimg.imsave(output_dir / f"{stem}_class_map.png", colourise(class_map))

    shares = class_shares(class_map, num_classes)
    totals += shares
    forest = float(shares[list(data_lib.FOREST_CLASS_INDICES)].sum())
    rows.append({
        "tile": stem,
        "height": class_map.shape[0],
        "width": class_map.shape[1],
        "mean_confidence": float(confidence.mean()),
        "forest_share": forest,
        **{f"share_{name.lower().replace(' ', '_')}": float(value)
           for name, value in zip(data_lib.CLASS_NAMES, shares)},
    })
    print(f"  {stem}: forest {100 * forest:5.1f}% | "
          f"natural {100 * shares[data_lib.NATURAL_FOREST_INDEX]:4.1f}% | "
          f"planted {100 * shares[data_lib.PLANTED_FOREST_INDEX]:4.1f}% | "
          f"tree crops {100 * shares[data_lib.TREE_CROPS_INDEX]:4.1f}%")

  table_path = output_dir / "predictions.csv"
  with table_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

  mean_shares = totals / len(rows)
  print("\naverage composition over all tiles:")
  for name, value in zip(data_lib.CLASS_NAMES, mean_shares):
    print(f"  {name:18s} {100 * value:5.2f}%")
  print(f"\nclass maps and {os.path.basename(table_path)} written to {output_dir}")


if __name__ == "__main__":
  main()
