"""Checkpoint settings use explicit overrides without inheriting run services."""

from __future__ import annotations

import argparse
import inspect
import json

import pytest

from mamba_forest.checkpoint import add_model_arguments, resolve_model_arguments
from mamba_forest.model import MODEL_DEFAULTS, MambaForestSegmenter


def _arguments(tmp_path, *flags):
  parser = argparse.ArgumentParser()
  parser.add_argument("--weights", default=str(tmp_path / "best.weights.h5"))
  add_model_arguments(parser)
  return parser.parse_args(flags)


def _write_config(tmp_path, config, name="config.json"):
  path = tmp_path / name
  path.write_text(json.dumps(config), encoding="utf-8")
  return path


def test_missing_adjacent_config_preserves_exact_legacy_defaults(tmp_path):
  args = _arguments(tmp_path)
  assert args.d_model is None and args.dropout is None
  assert resolve_model_arguments(args) is args
  expected = {
      "d_model": 32, "dropout": 0.2, "d_state_temporal": 8,
      "d_state_spatial": 16, "bottleneck_channels": 512,
  }
  assert {name: getattr(args, name) for name in expected} == expected
  assert args.training_config_path is None
  parameters = inspect.signature(MambaForestSegmenter.__init__).parameters
  assert {name: parameters[name].default for name in MODEL_DEFAULTS} == MODEL_DEFAULTS


def test_adjacent_config_restores_architecture_only(tmp_path):
  architecture = {
      "d_model": 12, "dropout": 0.35, "d_state_temporal": 3,
      "d_state_spatial": 5, "bottleneck_channels": 24,
  }
  path = _write_config(tmp_path, {
      **architecture, "gcs_root": "gs://must-not-be-inherited",
      "no_mixed_precision": False, "seed": 99,
      "output_dir": "another-run", "unknown_setting": {"anything": True},
  })
  args = _arguments(tmp_path)
  args.gcs_root = "local-records"
  args.no_mixed_precision = True
  args.output_dir = "current-output"
  before_defaults = dict(MODEL_DEFAULTS)
  resolve_model_arguments(args)
  assert {name: getattr(args, name) for name in architecture} == architecture
  assert args.training_config_path == str(path)
  assert args.gcs_root == "local-records" and args.no_mixed_precision is True
  assert args.output_dir == "current-output"
  assert not hasattr(args, "seed") and not hasattr(args, "unknown_setting")
  assert MODEL_DEFAULTS == before_defaults


def test_explicit_config_precedes_adjacent_and_flags_precede_config(tmp_path):
  _write_config(tmp_path, {"d_model": 99, "dropout": 0.5})
  explicit = _write_config(tmp_path, {
      "d_model": False, "dropout": 0.3, "d_state_temporal": 7,
  }, name="chosen.json")
  args = _arguments(tmp_path, "--config", str(explicit), "--d-model", "16",
                    "--dropout", "0")
  resolve_model_arguments(args)
  assert args.d_model == 16 and args.dropout == 0.0
  assert args.d_state_temporal == 7
  assert args.d_state_spatial == 16 and args.bottleneck_channels == 512
  assert args.training_config_path == str(explicit)


def test_no_config_ignores_malformed_sidecar_and_keeps_flags(tmp_path):
  (tmp_path / "config.json").write_text("invalid JSON", encoding="utf-8")
  args = _arguments(tmp_path, "--no-config", "--d-model", "4")
  resolve_model_arguments(args)
  assert args.d_model == 4 and args.dropout == 0.2
  assert args.training_config_path is None


def test_config_source_flags_are_mutually_exclusive(tmp_path):
  with pytest.raises(SystemExit) as error:
    _arguments(tmp_path, "--config", "chosen.json", "--no-config")
  assert error.value.code == 2


def test_explicit_missing_config_raises_with_path(tmp_path):
  path = tmp_path / "missing.json"
  with pytest.raises(FileNotFoundError, match="missing.json"):
    resolve_model_arguments(_arguments(tmp_path, "--config", str(path)))


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("payload", ["{broken", '{"dropout": NaN}',
                                     '{"d_model": Infinity}'])
def test_malformed_config_never_falls_back_silently(tmp_path, explicit, payload):
  path = tmp_path / "config.json"
  path.write_text(payload, encoding="utf-8")
  flags = ("--config", str(path)) if explicit else ()
  with pytest.raises(ValueError, match="Invalid JSON.*config.json"):
    resolve_model_arguments(_arguments(tmp_path, *flags))


@pytest.mark.parametrize("value", [[], None, 4, "settings"])
def test_config_must_be_flat_object(tmp_path, value):
  _write_config(tmp_path, value)
  with pytest.raises(ValueError, match="must be a JSON object"):
    resolve_model_arguments(_arguments(tmp_path))


def test_invalid_utf8_and_unreadable_file_type_fail_clearly(tmp_path):
  path = tmp_path / "config.json"
  path.write_bytes(b"\xff")
  with pytest.raises(ValueError, match="Invalid JSON.*config.json"):
    resolve_model_arguments(_arguments(tmp_path))
  with pytest.raises(IsADirectoryError):
    resolve_model_arguments(_arguments(tmp_path, "--config", str(tmp_path)))


@pytest.mark.parametrize("name", ["d_model", "d_state_temporal",
                                  "d_state_spatial", "bottleneck_channels"])
@pytest.mark.parametrize("value", [True, False, 0, -1, 2.5, "8", None])
def test_capacities_require_positive_integer_values(tmp_path, name, value):
  _write_config(tmp_path, {name: value})
  args = _arguments(tmp_path)
  with pytest.raises(ValueError, match=name + " must be a positive integer"):
    resolve_model_arguments(args)
  assert args.d_model is None  # Failed resolution does not partly mutate args.


@pytest.mark.parametrize("value", [-0.1, 1.0, "0.2", True, None])
def test_dropout_requires_finite_numeric_fraction(tmp_path, value):
  _write_config(tmp_path, {"dropout": value})
  with pytest.raises(ValueError, match=r"dropout.*finite.*\[0, 1\)"):
    resolve_model_arguments(_arguments(tmp_path))


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_cli_dropout_rejected(tmp_path, value):
  args = _arguments(tmp_path, "--dropout=" + value)
  with pytest.raises(ValueError, match="dropout must be a finite number"):
    resolve_model_arguments(args)


def test_model_variant_defaults_for_missing_and_legacy_configs(tmp_path):
  assert resolve_model_arguments(_arguments(tmp_path)).model_variant == "cnn_ssm"
  _write_config(tmp_path, {"d_model": 4})
  args = resolve_model_arguments(_arguments(tmp_path))
  assert args.model_variant == "cnn_ssm" and args.d_model == 4


def test_model_variant_restores_checkpoint_and_explicit_override(tmp_path):
  _write_config(tmp_path, {"model_variant": "cnn_only"})
  assert resolve_model_arguments(_arguments(tmp_path)).model_variant == "cnn_only"
  args = _arguments(tmp_path, "--model-variant", "cnn_ssm")
  assert resolve_model_arguments(args).model_variant == "cnn_ssm"
  explicit = _write_config(tmp_path, {"model_variant": "cnn_ssm"}, "chosen.json")
  args = _arguments(tmp_path, "--config", str(explicit))
  assert resolve_model_arguments(args).model_variant == "cnn_ssm"


def test_no_config_does_not_inherit_cnn_only_variant(tmp_path):
  _write_config(tmp_path, {"model_variant": "cnn_only"})
  args = _arguments(tmp_path, "--no-config")
  assert resolve_model_arguments(args).model_variant == "cnn_ssm"
  args = _arguments(tmp_path, "--no-config", "--model-variant", "cnn_only")
  assert resolve_model_arguments(args).model_variant == "cnn_only"


@pytest.mark.parametrize("value", ["unknown", "", None, True, 4, ["cnn_only"]])
def test_invalid_checkpoint_model_variant_fails_atomically(tmp_path, value):
  _write_config(tmp_path, {"model_variant": value, "d_model": 4})
  args = _arguments(tmp_path)
  with pytest.raises(ValueError, match="model_variant"):
    resolve_model_arguments(args)
  assert args.d_model is None and args.model_variant is None


def test_explicit_model_variant_overrides_invalid_saved_value(tmp_path):
  _write_config(tmp_path, {"model_variant": "invalid"})
  args = _arguments(tmp_path, "--model-variant", "cnn_only")
  assert resolve_model_arguments(args).model_variant == "cnn_only"


def test_unknown_model_variant_cli_value_is_rejected(tmp_path):
  with pytest.raises(SystemExit) as error:
    _arguments(tmp_path, "--model-variant", "not-a-model")
  assert error.value.code == 2
