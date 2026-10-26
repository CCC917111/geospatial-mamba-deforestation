"""Synthetic timings retain samples and exercise completed train/infer steps."""

from __future__ import annotations

import contextlib
import hashlib
import json
from types import SimpleNamespace

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


@pytest.mark.parametrize("model_variant", ["cnn_ssm", "cnn_only"])
def test_benchmark_train_and_infer_with_metadata(model_variant):
  args = benchmark.parse_args([
      "--mode", "both", "--tile-size", "8", "--batch-size", "1",
      "--d-model", "4", "--bottleneck-channels", "8", "--steps", "2",
      "--warmup", "1", "--device", "/CPU:0", "--model-variant", model_variant])
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
  assert metadata["model_config"]["model_variant"] == model_variant
  assert metadata["cpu_count"] is None or metadata["cpu_count"] > 0
  assert "TF_NUM_INTRAOP_THREADS" in metadata["thread_settings"]
  assert set(metadata["tensorflow_threading"]) == {
      "intra_op_parallelism_threads", "inter_op_parallelism_threads"}
  assert "No accuracy measurement" in metadata["scope"]
  assert metadata["soft_device_placement"] is False
  assert "CPU:0" in metadata["probe_device"]
  assert metadata["source"]["module_sha256"]["benchmark.py"]
  assert result["results"]["training_graph"]["optimizer_iterations"][
      "measured_updates"] == 2
  assert result["results"]["training_graph"]["memory"]["status"] == "not_applicable"
  json.dumps(result)


def test_requested_missing_gpu_rejects_cpu_fallback_before_model_creation(monkeypatch):
  original = tf.config.list_logical_devices
  monkeypatch.setattr(tf.config, "list_logical_devices",
                      lambda kind=None: [] if kind == "GPU" else original(kind))
  monkeypatch.setattr(benchmark, "MambaForestSegmenter",
                      lambda **kwargs: pytest.fail("model must not be built"))
  args = benchmark.parse_args(["--device", "/GPU:0"])
  previous_policy = tf.keras.mixed_precision.global_policy().name
  previous_placement = tf.config.get_soft_device_placement()
  with pytest.raises(ValueError, match="Requested device is unavailable"):
    benchmark.run_benchmark(args)
  assert tf.keras.mixed_precision.global_policy().name == previous_policy
  assert tf.config.get_soft_device_placement() == previous_placement


def test_strict_probe_rejects_wrong_actual_device_and_restores_policy(monkeypatch):
  previous = tf.config.get_soft_device_placement()
  monkeypatch.setattr(tf.config, "list_logical_devices", lambda kind: [
      SimpleNamespace(name="/device:GPU:0")])
  monkeypatch.setattr(tf, "device", lambda name: contextlib.nullcontext())
  monkeypatch.setattr(tf, "matmul", lambda *args: SimpleNamespace(
      device="/device:CPU:0", numpy=lambda: np.ones((2, 2))))
  with pytest.raises(RuntimeError, match="Device probe ran on"):
    with benchmark.strict_device("/GPU:0"):
      pytest.fail("wrong-device probe must reject entry")
  assert tf.config.get_soft_device_placement() == previous


def test_strict_device_restores_policy_when_measured_work_fails():
  previous = tf.config.get_soft_device_placement()
  with pytest.raises(RuntimeError, match="deliberate failure"):
    with benchmark.strict_device("/CPU:0"):
      assert tf.config.get_soft_device_placement() is False
      raise RuntimeError("deliberate failure")
  assert tf.config.get_soft_device_placement() == previous


def test_memory_peak_resets_after_warmup_and_preserves_live_baseline(monkeypatch):
  events = []
  snapshots = iter(({"current": 100, "peak": 900},
                    {"current": 120, "peak": 150}))

  def memory(device):
    events.append("memory")
    return next(snapshots)

  def step():
    events.append("step")
    return tf.constant(1.0)

  monkeypatch.setattr(tf.config.experimental, "get_memory_info", memory)
  monkeypatch.setattr(tf.config.experimental, "reset_memory_stats",
                      lambda device: events.append("reset"))
  result = benchmark.measure(step, 2, 3, 1, memory_device="GPU:0")
  assert events == ["step"] * 3 + ["memory", "reset", "step", "step", "memory"]
  assert result["memory"]["starting_current_bytes"] == 100
  assert result["memory"]["ending_current_bytes"] == 120
  assert result["memory"]["peak_bytes"] == 150
  assert result["memory"]["peak_above_starting_current_bytes"] == 50


def test_unavailable_allocator_statistics_are_not_reported_as_zero(monkeypatch):
  def unavailable(device):
    raise ValueError("allocator statistics unavailable")

  monkeypatch.setattr(tf.config.experimental, "get_memory_info", unavailable)
  result = benchmark.measure(lambda: tf.constant(1.0), 1, 1, 1,
                             memory_device="GPU:0")
  assert result["memory"]["status"] == "unavailable"
  assert "peak_bytes" not in result["memory"]
  assert result["measured_steps"] == 1


def test_training_update_checks_allow_warmup_skips_but_reject_measured_skips():
  calls = 0

  def adapted():
    nonlocal calls
    calls += 1
    # Simulate an initial loss-scale reduction without an optimizer update.
    return tf.constant(1.0), tf.constant(max(0, calls - 1))

  result = benchmark.measure(adapted, 2, 1, 1, iteration_output_index=-1)
  assert result["optimizer_iterations"] == {
      "after_warmup": 0, "after_measurement": 2, "measured_updates": 2}
  with pytest.raises(RuntimeError, match="skipped an optimizer update"):
    benchmark.measure(lambda: (tf.constant(1.0), tf.constant(0)),
                      1, 1, 1, iteration_output_index=-1)


def test_training_only_builds_one_model_and_loads_shared_checkpoint(
      tmp_path, monkeypatch):
  created = []

  class TinyModel(tf.keras.Model):
    def __init__(self, **kwargs):
      super().__init__()
      self.head = tf.keras.layers.Dense(kwargs["num_classes"], activation="softmax")
      created.append(self)

    def call(self, inputs, training=False):
      return self.head(inputs["s2"][:, 0])

  inputs, _ = benchmark.synthetic_batch(1, 8)
  fixture = TinyModel(num_classes=9)
  fixture(inputs)
  path = tmp_path / "shared.weights.h5"
  fixture.save_weights(path)
  created.clear()
  monkeypatch.setattr(benchmark, "MambaForestSegmenter", TinyModel)
  args = benchmark.parse_args([
      "--mode", "training", "--tile-size", "8", "--warmup", "1", "--steps", "1",
      "--device", "/CPU:0", "--weights", str(path)])
  result = benchmark.run_benchmark(args)
  assert len(created) == 1
  assert set(result["results"]) == {"training_graph"}
  assert result["metadata"]["weights_sha256"] == hashlib.sha256(
      path.read_bytes()).hexdigest()
  assert result["metadata"]["parameters"] == fixture.count_params()
