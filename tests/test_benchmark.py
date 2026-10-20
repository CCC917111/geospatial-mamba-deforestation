"""Synthetic timings retain samples and exercise completed train/infer steps."""

from __future__ import annotations

import json

import numpy as np
import pytest
import tensorflow as tf

from mamba_forest import benchmark


def test_latency_summary_uses_total_elapsed_for_throughput():
  result = benchmark.summarize_latencies([1.0, 2.0, 3.0], batch_size=4)
  assert result["latencies_seconds"] == [1.0, 2.0, 3.0]
  assert result["median_seconds"] == 2.0
  assert result["p95_seconds"] == pytest.approx(2.9)
  assert result["tiles_per_second"] == 2.0
  with pytest.raises(ValueError):
    benchmark.summarize_latencies([0.0], 1)


def test_measure_excludes_warmup_and_rejects_invalid_outputs():
  counter = tf.Variable(0)

  @tf.function
  def step():
    return counter.assign_add(1)

  result = benchmark.measure(step, steps=2, warmup=3, batch_size=1)
  assert int(counter.numpy()) == 5
  assert result["measured_steps"] == 2
  assert len(result["latencies_seconds"]) == 2
  with pytest.raises(FloatingPointError):
    benchmark.measure(lambda: tf.constant(np.nan), 1, 1, 1)


def test_benchmark_train_and_infer_with_metadata():
  args = benchmark.parse_args([
      "--mode", "both", "--tile-size", "8", "--batch-size", "1",
      "--d-model", "4", "--bottleneck-channels", "8", "--steps", "2",
      "--warmup", "1", "--device", "/CPU:0"])
  policy = tf.keras.mixed_precision.global_policy().name
  result = benchmark.run_benchmark(args)
  assert tf.keras.mixed_precision.global_policy().name == policy
  assert set(result["results"]) == {
      "inference_eager", "inference_graph", "training_graph"}
  for metrics in result["results"].values():
    assert metrics["measured_steps"] == 2
    assert metrics["tiles_per_second"] > 0
    assert metrics["p95_seconds"] >= metrics["median_seconds"] > 0
  metadata = result["metadata"]
  assert metadata["input_shapes"]["s2"] == [1, 4, 8, 8, 10]
  assert metadata["precision"] == "float32"
  assert metadata["parameters"] > 0
  assert metadata["cpu_count"] is None or metadata["cpu_count"] > 0
  assert "TF_NUM_INTRAOP_THREADS" in metadata["thread_settings"]
  assert set(metadata["tensorflow_threading"]) == {
      "intra_op_parallelism_threads", "inter_op_parallelism_threads"}
  assert "No accuracy measurement" in metadata["scope"]
  json.dumps(result)
