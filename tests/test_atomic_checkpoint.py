"""Interrupted checkpoint writes preserve the previous complete weights."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from mamba_forest import checkpoint


def test_completed_checkpoint_replaces_destination_and_returns_its_hash(tmp_path):
  destination = tmp_path / "best.weights.h5"
  destination.write_bytes(b"previous complete checkpoint")
  temporary_paths = []

  class Model:
    def save_weights(self, path):
      temporary = Path(path)
      assert temporary.parent == destination.parent
      assert temporary != destination and temporary.name.endswith(".weights.h5")
      assert destination.read_bytes() == b"previous complete checkpoint"
      temporary_paths.append(temporary)
      temporary.write_bytes(b"new complete checkpoint")

  digest = checkpoint.save_weights_atomic(Model(), destination)
  assert destination.read_bytes() == b"new complete checkpoint"
  assert digest == hashlib.sha256(destination.read_bytes()).hexdigest()
  assert not temporary_paths[0].exists()
  assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_partial_save_failure_preserves_previous_checkpoint(tmp_path, failure):
  destination = tmp_path / "best.weights.h5"
  destination.write_bytes(b"previous complete checkpoint")

  class Model:
    def save_weights(self, path):
      Path(path).write_bytes(b"partial write")
      raise failure("simulated interruption")

  with pytest.raises(failure, match="simulated interruption"):
    checkpoint.save_weights_atomic(Model(), destination)
  assert destination.read_bytes() == b"previous complete checkpoint"
  assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("phase", ["hash", "replace"])
def test_failed_finalization_cleans_temporary_weights(tmp_path, monkeypatch, phase):
  destination = tmp_path / "last.weights.h5"
  destination.write_bytes(b"previous complete checkpoint")

  class Model:
    def save_weights(self, path):
      Path(path).write_bytes(b"new complete checkpoint")

  def fail(*args):
    raise OSError("simulated " + phase + " failure")

  if phase == "hash":
    monkeypatch.setattr(checkpoint, "weights_sha256", fail)
  else:
    monkeypatch.setattr(checkpoint.os, "replace", fail)
  with pytest.raises(OSError, match="simulated " + phase + " failure"):
    checkpoint.save_weights_atomic(Model(), destination)
  assert destination.read_bytes() == b"previous complete checkpoint"
  assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("name", ["model.h5", "model.keras", "weights", "weights.h5"])
def test_invalid_h5_contract_fails_before_saving(tmp_path, name):
  class Model:
    def save_weights(self, path):
      pytest.fail("invalid checkpoint path reached model.save_weights")

  with pytest.raises(ValueError, match=r"end with \.weights\.h5"):
    checkpoint.save_weights_atomic(Model(), tmp_path / name)
  assert not list(tmp_path.iterdir())


def test_empty_save_does_not_replace_previous_checkpoint(tmp_path):
  destination = tmp_path / "best.weights.h5"
  destination.write_bytes(b"previous complete checkpoint")

  class Model:
    def save_weights(self, path):
      pass

  with pytest.raises(ValueError, match="nonempty weights file"):
    checkpoint.save_weights_atomic(Model(), destination)
  assert destination.read_bytes() == b"previous complete checkpoint"
  assert list(tmp_path.iterdir()) == [destination]


def test_real_keras_h5_checkpoint_loads_after_atomic_replacement(tmp_path):
  import tensorflow as tf

  model = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(3,)),
      tf.keras.layers.Dense(2, kernel_initializer="ones", bias_initializer="zeros"),
  ])
  inputs = tf.constant([[1.0, 2.0, 3.0]])
  expected = model(inputs).numpy()
  destination = tmp_path / "best.weights.h5"
  digest = checkpoint.save_weights_atomic(model, destination)
  model.set_weights([np.zeros((3, 2), np.float32), np.ones((2,), np.float32)])
  assert not np.allclose(model(inputs).numpy(), expected)
  model.load_weights(destination)
  np.testing.assert_array_equal(model(inputs).numpy(), expected)
  assert digest == checkpoint.weights_sha256(destination)
  assert list(tmp_path.iterdir()) == [destination]


def test_checkpoint_helpers_do_not_import_tensorflow():
  source = Path(checkpoint.__file__).resolve().parents[1]
  subprocess.run([
      sys.executable, "-c",
      "import sys; from mamba_forest.checkpoint import save_weights_atomic; "
      "assert 'tensorflow' not in sys.modules",
  ], env={**os.environ, "PYTHONPATH": str(source)}, check=True)
