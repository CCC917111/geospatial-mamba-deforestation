#!/usr/bin/env python3
"""Plot the per-epoch metrics written by mamba_forest.train.

Example:
    python scripts/plot_results.py results/training_results.csv \
        --output results/training_curves.png
"""

from __future__ import annotations

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend set above)


def read_csv(path: str) -> dict[str, list[float]]:
  with open(path, newline="") as f:
    rows = list(csv.DictReader(f))
  return {key: [float(row[key]) for row in rows] for key in rows[0]}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("csv_path", nargs="?", default="results/training_results.csv")
  parser.add_argument("--output", default="results/training_curves.png")
  args = parser.parse_args()

  data = read_csv(args.csv_path)
  epochs = data["epoch"]

  fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
  left.plot(epochs, data["train_loss"], marker="o", color="#444444")
  left.set_title("Training loss")
  left.set_xlabel("epoch")
  left.set_xticks(epochs)
  left.grid(alpha=0.3)

  for key, label in (("val_accuracy", "pixel accuracy"),
                     ("val_mean_iou", "mean IoU"),
                     ("macro_f1", "macro F1"),
                     ("forest_f1", "forest F1")):
    if key in data:
      right.plot(epochs, data[key], marker="o", label=label)
  right.set_title("Validation metrics")
  right.set_xlabel("epoch")
  right.set_xticks(epochs)
  right.set_ylim(0, 1)
  right.legend()
  right.grid(alpha=0.3)

  fig.tight_layout()
  fig.savefig(args.output, dpi=150)
  print(f"wrote {args.output}")


if __name__ == "__main__":
  main()
