"""Compute precision respects visible devices and explicit experiment settings."""

from __future__ import annotations

import sys

import numpy as np
import pytest
import tensorflow as tf

from mamba_forest import train


@pytest.fixture(autouse=True)
def restore_precision_policy():
  previous = tf.keras.mixed_precision.global_policy()
  yield
  tf.keras.mixed_precision.set_global_policy(previous)


@pytest.mark.parametrize("visible_gpu", [False, True])
@pytest.mark.parametrize("requested", ["auto", "float32", "mixed_float16"])
def test_selected_policy_controls_compute_and_keeps_float32_weights(
      monkeypatch, visible_gpu, requested):
  # Device discovery is mocked, not GPU execution. The dense operation runs
  # on the available local hardware under the selected compute policy.
  monkeypatch.setattr(tf.config, "get_visible_devices",
                      lambda kind: [object()] if visible_gpu else [])
  expected = requested
  if requested == "auto":
    expected = "mixed_float16" if visible_gpu else "float32"
  resolved = train.configure_precision(requested)
  layer = tf.keras.layers.Dense(2)
  outputs = layer(tf.ones((1, 3), dtype=tf.float32))
  assert resolved == expected
  assert outputs.dtype == (tf.float16 if expected == "mixed_float16" else tf.float32)
  assert all(tf.as_dtype(variable.dtype) == tf.float32 for variable in layer.weights)
  assert np.isfinite(outputs.numpy()).all()


def test_cpu_auto_resets_an_inherited_mixed_policy(monkeypatch):
  monkeypatch.setattr(tf.config, "get_visible_devices", lambda kind: [])
  tf.keras.mixed_precision.set_global_policy("mixed_float16")
  assert train.configure_precision() == "float32"
  assert tf.keras.mixed_precision.global_policy().name == "float32"


@pytest.mark.parametrize("requested", ["auto", "float32"])
def test_legacy_float32_flag_overrides_auto_without_requiring_hardware(
      monkeypatch, requested):
  monkeypatch.setattr(tf.config, "get_visible_devices", lambda kind: [object()])
  assert train.configure_precision(requested, no_mixed_precision=True) == "float32"


def test_conflicting_precision_flags_fail_before_data_io(tmp_path, monkeypatch):
  opened = []
  monkeypatch.setattr(train.data_lib, "make_dataset", lambda *a, **k: opened.append(a))
  monkeypatch.setattr(sys, "argv", [
      "mamba_forest.train", "--precision", "mixed_float16",
      "--no-mixed-precision", "--output-dir", str(tmp_path / "run"),
  ])
  with pytest.raises(SystemExit) as error:
    train.main()
  assert error.value.code == 2
  assert not opened
  assert not (tmp_path / "run").exists()
