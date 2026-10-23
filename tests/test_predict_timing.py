"""Production prediction timings cover real file IO and requested host copies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json

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
