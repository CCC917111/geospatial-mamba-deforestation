"""Prediction validates raw tiles and preserves eager output in graph mode."""

from __future__ import annotations

import csv
import json
import sys

import numpy as np
import pytest
import tensorflow as tf

from mamba_forest import predict
from mamba_forest.model import MambaForestSegmenter


def _tile(size=8, time=4):
  rng = np.random.default_rng(42)
  return {
      "s2": rng.uniform(0, 10000, (time, size, size, 10)).astype(np.float32),
      "elevation": rng.uniform(0, 100, (size, size, 3)).astype(np.float32),
      "climate": rng.normal(size=(time, 14)).astype(np.float32),
  }


@pytest.mark.parametrize("issue", ["time", "empty", "nan", "missing"])
def test_invalid_tiles_rejected(tmp_path, issue):
  tile = _tile()
  if issue == "time":
    tile = _tile(time=3)
  elif issue == "empty":
    tile = _tile(size=0)
  elif issue == "nan":
    tile["climate"][0, 0] = np.nan
  else:
    del tile["elevation"]
  path = tmp_path / "invalid.npz"
  np.savez(path, **tile)
  with pytest.raises(ValueError):
    predict.load_tile(path)


def test_graph_prediction_agrees_with_eager_without_retracing_sizes():
  tf.keras.utils.set_random_seed(42)
  model = MambaForestSegmenter(9, d_model=4, bottleneck_channels=8)
  model(predict.preprocess_tile(_tile()), training=False)
  predictor = predict.make_predictor(model, save_probabilities=True)
  for size in (8, 16):
    inputs = predict.preprocess_tile(_tile(size=size))
    expected = model(inputs, training=False).numpy()
    outputs = predictor(inputs)
    np.testing.assert_allclose(outputs["probabilities"].numpy(), expected,
                               rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(outputs["class_map"].numpy(),
                                  expected.argmax(axis=-1))
    np.testing.assert_allclose(outputs["mean_confidence"].numpy(),
                               expected.max(axis=-1).mean(axis=(1, 2)),
                               rtol=2e-5)
  assert predictor.experimental_get_tracing_count() == 1
  compact = predict.make_predictor(model)(inputs)
  assert set(compact) == {"class_map", "mean_confidence"}
  assert compact["class_map"].dtype == tf.uint8


@pytest.mark.parametrize("configuration", ["sidecar", "manual", "moved_config"])
def test_prediction_cli_saves_consistent_outputs(
        tmp_path, monkeypatch, configuration):
  tf.keras.utils.set_random_seed(42)
  tiles = tmp_path / "tiles"
  tiles.mkdir()
  tile = _tile()
  np.savez(tiles / "tile.npz", **tile)
  architecture = {
      "d_model": 4, "dropout": 0.1, "d_state_temporal": 2,
      "d_state_spatial": 3, "bottleneck_channels": 8,
  }
  model = MambaForestSegmenter(
      9, d_model=4, dropout_rate=0.1, d_state_temporal=2,
      d_state_spatial=3, bottleneck_channels=8)
  expected = model(predict.preprocess_tile(tile), training=False).numpy()[0]
  weights_dir = tmp_path / "moved" if configuration == "moved_config" else tmp_path
  weights_dir.mkdir(exist_ok=True)
  weights = weights_dir / "test.weights.h5"
  model.save_weights(weights)
  configuration_flags = []
  if configuration == "manual":
    for name, value in architecture.items():
      configuration_flags.extend(["--" + name.replace("_", "-"), str(value)])
  else:
    # Training's flat config may contain unrelated optimizer/data settings.
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({**architecture, "learning_rate": 2e-4}),
                           encoding="utf-8")
    if configuration == "moved_config":
      configuration_flags = ["--config", str(config_path)]
  output = tmp_path / "predictions"
  monkeypatch.setattr(sys, "argv", [
      "predict", "--weights", str(weights), "--input-dir", str(tiles),
      "--output-dir", str(output), "--save-probabilities", "--save-png",
      *configuration_flags])
  predict.main()
  probabilities = np.load(output / "tile_probabilities.npy")
  class_map = np.load(output / "tile_class_map.npy")
  np.testing.assert_allclose(probabilities, expected, rtol=2e-5, atol=2e-6)
  np.testing.assert_array_equal(class_map, probabilities.argmax(axis=-1))
  assert (output / "tile_class_map.png").is_file()
  with (output / "predictions.csv").open() as handle:
    rows = list(csv.DictReader(handle))
  assert len(rows) == 1 and rows[0]["tile"] == "tile"
  np.testing.assert_allclose(float(rows[0]["mean_confidence"]),
                             probabilities.max(axis=-1).mean(), rtol=2e-5)


@pytest.mark.parametrize("skip_config", [False, True])
def test_prediction_without_config_preserves_legacy_defaults(
        tmp_path, monkeypatch, skip_config):
  if skip_config:
    (tmp_path / "config.json").write_text("malformed", encoding="utf-8")
  monkeypatch.setattr(sys, "argv", [
      "predict", "--weights", str(tmp_path / "bare.weights.h5"),
      "--input-dir", str(tmp_path / "tiles"),
      "--output-dir", str(tmp_path / "outputs"),
      *(["--no-config"] if skip_config else []),
  ])
  args = predict.parse_args()
  assert (args.d_model, args.dropout, args.d_state_temporal,
          args.d_state_spatial, args.bottleneck_channels) == (32, 0.2, 8, 16, 512)
  assert args.training_config_path is None


@pytest.mark.parametrize("issue", ["malformed", "missing"])
def test_bad_configuration_fails_before_tile_access(
        tmp_path, monkeypatch, issue):
  config = tmp_path / "config.json"
  if issue == "malformed":
    config.write_text("{invalid", encoding="utf-8")
  output = tmp_path / "outputs"
  monkeypatch.setattr(sys, "argv", [
      "predict", "--weights", str(tmp_path / "bare.weights.h5"),
      "--input-dir", str(tmp_path / "tiles"), "--output-dir", str(output),
      "--config", str(config),
  ])

  def unexpected_tile_access(*args, **kwargs):
    pytest.fail("invalid model configuration must fail before accessing tiles")

  monkeypatch.setattr(predict.Path, "glob", unexpected_tile_access)
  monkeypatch.setattr(predict, "load_tile", unexpected_tile_access)
  with pytest.raises(SystemExit) as error:
    predict.main()
  assert error.value.code == 2
  assert not output.exists()
