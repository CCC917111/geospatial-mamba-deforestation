"""Bounded prediction batches preserve per-tile normalization and exports."""

from __future__ import annotations

import argparse
import csv
import sys
import weakref

import numpy as np
import pytest
import tensorflow as tf

from mamba_forest import benchmark_predict, predict
from mamba_forest.model import MambaForestSegmenter


def _tiles(directory):
  directory.mkdir()
  rng = np.random.default_rng(103)
  tiles = []
  for index, size in enumerate((8, 8, 8, 16, 8, 8, 8)):
    tile = {
        "s2": rng.uniform(0, 10000, (4, size, size, 10)).astype(np.float32),
        "elevation": rng.uniform(0, 100, (size, size, 3)).astype(np.float32),
        "climate": (rng.normal(size=(4, 14)) * (index + 1)
                    + np.arange(4)[:, None] ** 2 + index * 70).astype(np.float32),
    }
    np.savez(directory / f"tile_{index}.npz", **tile)
    tiles.append(tile)
  return tiles


def _arguments(tmp_path, directory, weights, batch_size):
  return argparse.Namespace(
      weights=str(weights), input_dir=str(directory),
      output_dir=tmp_path / f"outputs-b{batch_size}", pattern="*.npz",
      batch_size=batch_size, save_png=True, save_probabilities=True,
      d_model=4, dropout=0.0, d_state_temporal=2, d_state_spatial=2,
      bottleneck_channels=8, training_config_path=None,
      repeats=2, warmup=0)


def test_real_checkpoint_batches_keep_order_shapes_climate_and_timing(
        tmp_path, monkeypatch):
  directory = tmp_path / "tiles"
  tiles = _tiles(directory)
  tf.keras.utils.set_random_seed(103)
  model = MambaForestSegmenter(
      9, d_model=4, dropout_rate=0.0, d_state_temporal=2, d_state_spatial=2,
      bottleneck_channels=8)
  model(predict.preprocess_tile(tiles[0]), training=False)
  weights = tmp_path / "fixture.weights.h5"
  model.save_weights(weights)
  single_args = _arguments(tmp_path, directory, weights, 1)
  # Existing API callers may omit the newly optional setting.
  del single_args.batch_size
  single = predict.run_prediction(single_args, quiet=True)
  assert single["metadata"]["requested_batch_size"] == 1
  assert len(single["batches"]) == len(tiles)

  original_load, original_read = predict.load_tile, predict._read_inputs
  original_predictor = predict.make_predictor
  reads, calls = [], []
  normalized_refs = []
  processed = 0

  def load(path):
    reads.append(path.name)
    return original_load(path)

  def read_inputs(path):
    if path.stem == "tile_2":
      # The first full B2 group has already been exported and released.
      # Keras may capture the first build example in a FuncGraph; all later
      # tile inputs must be released. A last-tile loop-local leak fails here.
      owned_refs = normalized_refs[len(predict.REQUIRED_ARRAYS):]
      assert not any(reference() is not None for reference in owned_refs)
    prepared = original_read(path)
    normalized_refs.extend(weakref.ref(value) for value in prepared[0].values())
    return prepared

  def make_predictor(model, save_probabilities=False):
    infer = original_predictor(model, save_probabilities)

    def run(inputs):
      nonlocal processed
      count = int(inputs["climate"].shape[0])
      # A same-shape full batch needs no lookahead. Shape changes may retain
      # only one additional prepared tile, rather than staging the directory.
      assert len(reads) - processed <= 3
      calls.append(inputs["climate"].numpy())
      processed += count
      return infer(inputs)
    return run

  monkeypatch.setattr(predict, "load_tile", load)
  monkeypatch.setattr(predict, "_read_inputs", read_inputs)
  monkeypatch.setattr(predict, "make_predictor", make_predictor)
  batch_args = _arguments(tmp_path, directory, weights, 2)
  report = benchmark_predict.run_benchmark(batch_args)
  assert reads == [f"tile_{i}.npz" for _ in range(2) for i in range(7)]
  assert [len(value) for value in calls] == [2, 1, 1, 2, 1] * 2
  expected = [predict.preprocess_tile(tile)["climate"].numpy()[0] for tile in tiles]
  for pass_index in range(2):
    observed = np.concatenate(calls[pass_index * 5:(pass_index + 1) * 5])
    np.testing.assert_allclose(observed, np.stack(expected), rtol=1e-6, atol=1e-6)
  assert report["requested_batch_size"] == 2
  assert report["schema_version"] == 2
  assert report["metadata"]["model_trainable_variable_devices"]
  output_devices = report["metadata"]["first_predictor_output_devices"]
  assert set(output_devices) == {"class_map", "mean_confidence", "probabilities"}
  assert all(value.startswith("/") for value in output_devices.values())
  assert report["measured_tile_count"] == 14
  assert report["steady_state_tile_count"] == 12
  assert report["measured_batch_count"] == 10
  assert report["steady_state_batch_count"] == 9
  assert report["measured_tiles_per_second"] == pytest.approx(
      14 / report["measured_pass_wall_seconds"])
  assert report["first_inference_sync_seconds"] == (
      report["batches"][0]["inference_sync_seconds"])
  batch_times = report["steady_state_batch_stages"]["inference_sync_seconds"]["seconds"]
  assert len(batch_times) == 9
  for batch in report["batches"]:
    samples = [sample for sample in report["samples"]
               if sample["phase"] == batch["phase"]
               and sample["pass_index"] == batch["pass_index"]
               and sample["batch_index"] == batch["batch_index"]]
    assert len(samples) == batch["batch_size"]
    assert sum(s["inference_sync_seconds"] for s in samples) == pytest.approx(
        batch["inference_sync_seconds"])
    assert sum(s["batch_assembly_share_seconds"] for s in samples) == pytest.approx(
        batch["batch_assembly_seconds"])
    assert all(s["first_inference"] == batch["first_inference"] for s in samples)
  first_samples = [sample for sample in report["samples"] if sample["first_inference"]]
  assert [sample["tile_index"] for sample in first_samples] == [0, 1]
  for pass_index in range(2):
    output = batch_args.output_dir / f"measured_{pass_index:03d}"
    with (output / "predictions.csv").open() as handle:
      rows = list(csv.DictReader(handle))
    assert [row["tile"] for row in rows] == [f"tile_{i}" for i in range(7)]
    for index, tile in enumerate(tiles):
      name = f"tile_{index}"
      assert (output / f"{name}_class_map.png").is_file()
      probabilities = np.load(output / f"{name}_probabilities.npy")
      reference = np.load(single_args.output_dir / f"{name}_probabilities.npy")
      np.testing.assert_allclose(probabilities, reference, rtol=2e-5, atol=2e-6)
      class_map = np.load(output / f"{name}_class_map.npy")
      np.testing.assert_array_equal(class_map, probabilities.argmax(axis=-1))
      assert class_map.shape == tile["s2"].shape[1:3]
      assert int(rows[index]["height"]) == class_map.shape[0]
      assert int(rows[index]["width"]) == class_map.shape[1]
      assert float(rows[index]["mean_confidence"]) == pytest.approx(
          probabilities.max(axis=-1).mean(), rel=2e-5)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
@pytest.mark.parametrize("module", [predict, benchmark_predict])
def test_invalid_batch_api_fails_before_output_or_model(tmp_path, module, batch_size):
  output = tmp_path / "outputs"
  args = argparse.Namespace(batch_size=batch_size, output_dir=output)
  function = module.run_prediction if module is predict else module.run_benchmark
  with pytest.raises(ValueError, match="positive integer"):
    function(args)
  assert not output.exists()


@pytest.mark.parametrize("module", [predict, benchmark_predict])
@pytest.mark.parametrize("batch_size", ["0", "-1"])
def test_invalid_batch_cli_fails_before_output(
        tmp_path, monkeypatch, module, batch_size):
  output = tmp_path / "outputs"
  flags = ["--weights", "missing.weights.h5", "--input-dir", "missing",
           "--output-dir", str(output), "--batch-size", batch_size]
  if module is predict:
    monkeypatch.setattr(sys, "argv", ["predict", *flags])
  with pytest.raises(SystemExit):
    module.parse_args() if module is predict else module.parse_args(flags)
  assert not output.exists()
