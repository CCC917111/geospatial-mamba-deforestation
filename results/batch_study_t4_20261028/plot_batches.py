"""Regenerate the retained T4 throughput figure from the uploaded summary."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
RESULT = Path(__file__).with_name("study") / "study.json"


def main():
  data = json.loads(RESULT.read_text())
  plt.rcParams.update({
      "font.family": "DejaVu Sans", "font.size": 10,
      "axes.spines.top": False, "axes.spines.right": False,
      "svg.fonttype": "none", "svg.hashsalt": "batch-t4-20261028"})
  figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
  for axis, (mode, color) in zip(axes, [
      ("inference", "#176b49"), ("training", "#236a9a")]):
    cases = data["summary"][mode]["batches"]
    positions = range(len(cases))
    rates = [case["tiles_per_second"] for case in cases]
    low = [min(case["round_tiles_per_second"]) for case in cases]
    high = [max(case["round_tiles_per_second"]) for case in cases]
    axis.fill_between(positions, low, high, color=color, alpha=.13,
                      label="Three-round range")
    axis.plot(positions, rates, marker="o", markersize=6, lw=2.3, color=color,
              label="Pooled throughput")
    axis.set_xticks(list(positions), [str(case["batch_size"]) for case in cases])
    axis.set_xlabel("Batch size")
    axis.set_ylabel("Tiles / second")
    axis.set_ylim(0, max(high) * 1.2)
    axis.grid(axis="y", alpha=.18)
    axis.set_axisbelow(True)
    axis.set_title("FP32 inference" if mode == "inference" else
                   "Mixed FP16 training", fontweight="bold", loc="left")
    axis.annotate(f"{rates[-1] / rates[0]:.2f}× batch-one throughput",
                  (3, rates[-1]), xytext=(-9, 17), textcoords="offset points",
                  ha="right", color=color, fontweight="bold")
    axis.legend(loc="upper left", frameon=False, fontsize=8)
  figure.suptitle("Tesla T4: throughput increases with larger batches",
                  x=.08, ha="left", fontsize=15, fontweight="bold")
  figure.text(.08, .025,
              "One repeated synthetic tile · 3 alternating rounds × 30 steps · "
              "Explicit GPU scope · IO excluded", fontsize=9, color="#555555")
  figure.subplots_adjust(left=.08, right=.97, bottom=.17, top=.79, wspace=.27)
  directory = ROOT / "docs/figures"
  figure.savefig(directory / "t4-batch-scaling.svg", metadata={"Date": None})
  figure.savefig(directory / "t4-batch-scaling.png", dpi=160)


if __name__ == "__main__":
  main()
