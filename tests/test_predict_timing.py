"""Production prediction timings cover real file IO and requested host copies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from mamba_forest import predict


def _arguments(tmp_path):
  inputs = tmp_path / "tiles"
  inputs.mkdir()
  for index in range(2):
    np.savez(inputs / f"tile_{index}.npz",
             s2=np.zeros((4, 8, 8, 10), np.float32),
             elevation=np.zeros((8, 8, 3), np.float32),
             climate=np.zeros((4, 14), np.float32))
  weights = tmp_path / "fixture.weights.h5"
  weights.write_bytes(b"stub checkpoint identity")
  output = tmp_path / "predictions"
  return argparse.Namespace(
      weights=str(weights), input_dir=str(inputs), output_dir=str(output),
      pattern="*.npz", save_png=True, save_probabilities=True,
      d_model=4, dropout=0.0, d_state_temporal=2, d_state_spatial=2,
      bottleneck_channels=8, training_config_path=None,
      timings=output / "timings.json")


def _lightweight_predictor(monkeypatch):
  synchronized, reads = [], []
  original_load = predict.load_tile

  class Model:
    def __init__(self, **kwargs):
      pass

    def __call__(self, inputs, training=False):
      return tf.ones((1, 8, 8, 9)) / 9

    def load_weights(self, path):
      pass

  class HostArray:
    def __init__(self, key, value):
      self.key, self.value = key, value

    def numpy(self):
      synchronized.append(self.key)
      return self.value

  def make_predictor(model, save_probabilities=False):
    def infer(inputs):
      arrays = {
          "class_map": np.zeros((1, 8, 8), np.uint8),
          "mean_confidence": np.array([1 / 9], np.float32),
      }
      if save_probabilities:
        arrays["probabilities"] = np.full((1, 8, 8, 9), 1 / 9, np.float32)
      return {key: HostArray(key, value) for key, value in arrays.items()}
    return infer

  def load(path):
    reads.append(path)
    return original_load(path)

  monkeypatch.setattr(predict, "MambaForestSegmenter", Model)
  monkeypatch.setattr(predict, "make_predictor", make_predictor)
  monkeypatch.setattr(predict, "load_tile", load)
  return synchronized, reads


def test_default_outputs_and_raw_timings_include_actual_requested_copies(
      tmp_path, monkeypatch):
  args = _arguments(tmp_path)
  synchronized, reads = _lightweight_predictor(monkeypatch)
  report = predict.run_prediction(args, quiet=True)
  assert json.loads(args.timings.read_text()) == report
  assert len(reads) == len(report["samples"]) == 2
  assert synchronized.count("probabilities") == 2
  assert report["metadata"]["source_file_count"] == 2
  assert report["metadata"]["sources"][0]["shapes"]["s2"] == [4, 8, 8, 10]
  assert report["metadata"]["weights_sha256"] == hashlib.sha256(
      b"stub checkpoint identity").hexdigest()
  assert report["metadata"]["save_options"] == {"png": True, "probabilities": True}
  assert report["cold_start_to_first_prediction_seconds"] > 0
  assert report["samples"][0]["first_inference"] is True
  assert report["samples"][1]["first_inference"] is False
  stages = ("read_validate_seconds", "preprocess_seconds", "inference_sync_seconds",
            "postprocess_seconds", "array_image_export_seconds")
  for sample in report["samples"]:
    assert all(sample[key] >= 0 for key in stages)
    expected = sum(sample[key] for key in stages)
    assert sample["pipeline_seconds"] == pytest.approx(expected)
  assert report["passes"][0]["csv_export_seconds"] > 0
  assert report["total_wall_seconds"] >= report["model_build_load_seconds"]
  output = args.timings.parent
  for index in range(2):
    assert (output / f"tile_{index}_class_map.png").is_file()
    probabilities = np.load(output / f"tile_{index}_probabilities.npy")
    np.testing.assert_array_equal(np.load(output / f"tile_{index}_class_map.npy"),
                                  probabilities.argmax(axis=-1))
  with (output / "predictions.csv").open() as handle:
    assert len(list(csv.DictReader(handle))) == 2


def test_passes_reuse_model_but_read_every_tile_and_keep_outputs_separate(
      tmp_path, monkeypatch):
  args = _arguments(tmp_path)
  args.save_png = args.save_probabilities = False
  synchronized, reads = _lightweight_predictor(monkeypatch)
  report = predict.run_prediction(args, repeats=2, warmup=1, quiet=True)
  assert len(reads) == 6 and len(report["samples"]) == 6
  assert sum(sample["first_inference"] for sample in report["samples"]) == 1
  assert report["samples"][0]["phase"] == "warmup"
  assert [item["phase"] for item in report["passes"]] == [
      "warmup", "measured", "measured"]
  assert "probabilities" not in synchronized
  for name in ("warmup_000", "measured_000", "measured_001"):
    directory = args.timings.parent / name
    assert (directory / "predictions.csv").is_file()
    assert len(list(directory.glob("*_class_map.npy"))) == 2
  assert not list(args.timings.parent.glob("*_class_map.npy"))


def test_existing_timing_or_checkpoint_path_is_never_overwritten(tmp_path, monkeypatch):
  args = _arguments(tmp_path)
  original = args.weights
  args.timings = tmp_path / "config.json"
  args.timings.write_text("preserve me")
  with pytest.raises(ValueError, match="new .json file"):
    predict.run_prediction(args)
  assert args.timings.read_text() == "preserve me"
  assert not args.timings.parent.joinpath("predictions").exists()
  args.timings = original
  with pytest.raises(ValueError, match="new .json file"):
    predict.run_prediction(args)
  assert not args.timings == args.output_dir


def test_preprocessing_completes_before_its_timer_stops(tmp_path, monkeypatch):
  args = _arguments(tmp_path)
  synchronized = []
  original = predict._synchronize_preprocessing

  def synchronize(inputs):
    original(inputs)
    synchronized.append(True)

  calls = 0

  def clock():
    nonlocal calls
    calls += 1
    if calls == 4:
      assert synchronized, "preprocessing timer stopped before tensor completion"
    return float(calls)

  monkeypatch.setattr(predict, "_synchronize_preprocessing", synchronize)
  monkeypatch.setattr(predict.time, "perf_counter", clock)
  _, _, read_seconds, preprocess_seconds = predict._read_inputs(
      Path(args.input_dir) / "tile_0.npz")
  assert read_seconds == preprocess_seconds == 1.0
  assert calls == 4


def test_preprocessing_barrier_rejects_nonfinite_normalized_inputs():
  with pytest.raises(FloatingPointError, match="preprocessing"):
    predict._synchronize_preprocessing({
        "first": tf.constant([0.0]), "last": tf.constant([float("nan")])})


def test_checkpoint_change_during_loading_is_rejected(tmp_path, monkeypatch):
  args = _arguments(tmp_path)
  _lightweight_predictor(monkeypatch)
  original_model = predict.MambaForestSegmenter

  class MutatingModel(original_model):
    def load_weights(self, path):
      Path(path).write_bytes(b"checkpoint changed during load")

  monkeypatch.setattr(predict, "MambaForestSegmenter", MutatingModel)
  monkeypatch.setattr(predict, "make_predictor",
                      lambda *args: pytest.fail("must reject before prediction"))
  with pytest.raises(ValueError, match="Checkpoint changed while loading"):
    predict.run_prediction(args, quiet=True)


def test_report_identifies_loaded_checkpoint_even_if_file_changes_later(
      tmp_path, monkeypatch):
  args = _arguments(tmp_path)
  original_hash = hashlib.sha256(Path(args.weights).read_bytes()).hexdigest()
  _lightweight_predictor(monkeypatch)
  original_make = predict.make_predictor

  def make_predictor(model, save_probabilities=False):
    infer = original_make(model, save_probabilities)

    def predict_and_mutate(inputs):
      outputs = infer(inputs)
      Path(args.weights).write_bytes(b"later checkpoint replacement")
      return outputs

    return predict_and_mutate

  monkeypatch.setattr(predict, "make_predictor", make_predictor)
  report = predict.run_prediction(args, quiet=True)
  assert report["metadata"]["weights_sha256"] == original_hash
  assert hashlib.sha256(Path(args.weights).read_bytes()).hexdigest() != original_hash
