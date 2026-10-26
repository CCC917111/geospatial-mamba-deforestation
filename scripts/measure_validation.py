"""Isolate evaluation-graph reuse with the same model and in-memory batches."""
from __future__ import annotations
import argparse
import json
import statistics
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
from measure_revision import (configure_device, digest, make_model,
                              source_metadata, write_json)
from mamba_forest.train import DatasetEvaluator, evaluate


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--fixture-dir", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--device", default="/CPU:0")
  parser.add_argument("--steps", type=int, default=3)
  parser.add_argument("--batches", type=int, default=20)
  args = parser.parse_args()
  if min(args.steps, args.batches) < 1 or args.output.exists():
    parser.error("Positive budgets and a new output path are required")
  runtime = configure_device(tf, args.device)
  tf.keras.mixed_precision.set_global_policy("float32")
  fixture = json.loads((args.fixture_dir / "fixture.json").read_text())
  weights = args.fixture_dir / "initial.weights.h5"
  if digest(weights) != fixture["weights_sha256"]:
    raise ValueError("Shared weights changed")
  if digest(args.fixture_dir / "inputs.npz") != fixture["inputs_sha256"]:
    raise ValueError("Shared inputs changed")
  with tf.device(args.device):
    with np.load(args.fixture_dir / "inputs.npz", allow_pickle=False) as stored:
      inputs = {key: tf.constant(stored[key]) for key in ("s2", "elevation", "climate")}
      labels = tf.one_hot(stored["labels"], 9)
    model = make_model(tf, fixture["model_config"], inputs, weights)
    retained = DatasetEvaluator(model, 9)
  with tf.device("/CPU:0"):
    data = tf.data.Dataset.from_tensors(
        {**inputs, "labels": labels}).repeat(args.batches)
  callbacks = {"one_shot": lambda: evaluate(model, data, 9, return_confusion=True),
               "reused": lambda: retained(data, return_confusion=True)}
  records = {key: {"latencies_seconds": []} for key in callbacks}
  expected = None
  for key, callback in callbacks.items():
    started = time.perf_counter()
    result = callback()
    records[key]["first_pass_seconds"] = time.perf_counter() - started
    if expected is None:
      expected = result[-1]
    np.testing.assert_array_equal(result[-1], expected)
  traces = retained.accumulate.experimental_get_tracing_count()
  for index in range(args.steps):
    order = ("one_shot", "reused") if index % 2 == 0 else ("reused", "one_shot")
    for key in order:
      started = time.perf_counter()
      result = callbacks[key]()
      records[key]["latencies_seconds"].append(time.perf_counter() - started)
      np.testing.assert_array_equal(result[-1], expected)
  for record in records.values():
    record["median_seconds"] = statistics.median(record["latencies_seconds"])
  if retained.accumulate.experimental_get_tracing_count() != traces:
    raise RuntimeError("Retained evaluator retraced during repeated passes")
  write_json(args.output, {
      "status": "passed", "runtime": runtime, "results": records,
      "reused_trace_count": traces, "confusion_matrices_equal": True,
      "model_config": fixture["model_config"],
      "weights_sha256": fixture["weights_sha256"], "source": source_metadata(),
      "measurement_script_sha256": digest(__file__),
      "batches_per_pass": args.batches, "steps": args.steps,
      "speedup": (records["one_shot"]["median_seconds"]
                  / records["reused"]["median_seconds"]),
      "scope": "Same optimized model, one-shot versus retained evaluation graphs; "
               "includes dataset iteration, graph tracing and confusion metrics. "
               "Repeated synthetic batches; no remote data IO or accuracy claim."})
  print(args.output)


if __name__ == "__main__":
  main()
