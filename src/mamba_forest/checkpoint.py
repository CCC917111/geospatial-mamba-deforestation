"""Resolve checkpoint architecture settings without importing run data settings.

An explicit architecture flag overrides its flat training-config value, which
overrides the legacy model default. Only the architecture settings are
read; data paths, precision and all other training configuration stay separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
from pathlib import Path

from mamba_forest.constants import (DEFAULT_MODEL_VARIANT, MODEL_DEFAULTS,
                                    MODEL_VARIANTS)

_ARGUMENT_TO_MODEL_KEY = {
    "d_model": "d_model",
    "dropout": "dropout_rate",
    "d_state_temporal": "d_state_temporal",
    "d_state_spatial": "d_state_spatial",
    "bottleneck_channels": "bottleneck_channels",
}


def weights_sha256(path: str | Path) -> str:
  """Identifies checkpoint bytes without loading the whole file into memory."""
  digest = hashlib.sha256()
  with open(path, "rb") as checkpoint:
    for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
  """Adds optional architecture overrides and config-source selection flags."""
  for name in _ARGUMENT_TO_MODEL_KEY:
    parser.add_argument(
        "--" + name.replace("_", "-"), default=None,
        type=float if name == "dropout" else int,
        help="Override the checkpoint configuration's " + name.replace("_", " "))
  parser.add_argument("--model-variant", choices=MODEL_VARIANTS, default=None,
                      help="Override the checkpoint's CNN/SSM architecture variant")
  source = parser.add_mutually_exclusive_group()
  source.add_argument("--config", help="Read architecture from this training JSON")
  source.add_argument(
      "--no-config", action="store_true",
      help="Ignore adjacent config.json and use flags or legacy model defaults")


def _reject_json_constant(value):
  raise ValueError(f"non-finite JSON constant {value!r}")


def _read_config(args):
  explicit = getattr(args, "config", None)
  if getattr(args, "no_config", False):
    if explicit is not None:
      raise ValueError("--config and --no-config cannot be used together")
    return {}, None
  candidate = Path(explicit) if explicit is not None else (
      Path(args.weights).parent / "config.json")
  try:
    with candidate.open(encoding="utf-8") as handle:
      config = json.load(handle, parse_constant=_reject_json_constant)
  except FileNotFoundError:
    if explicit is not None:
      raise
    return {}, None
  except ValueError as error:
    raise ValueError(f"Invalid JSON in model config {candidate}: {error}") from error
  if not isinstance(config, dict):
    raise ValueError(f"Model config {candidate} must be a JSON object")
  return config, str(candidate)


def _validate(name, value):
  if name == "dropout":
    invalid = (isinstance(value, bool) or not isinstance(value, numbers.Real)
               or not math.isfinite(value) or not 0 <= value < 1)
    if invalid:
      raise ValueError("dropout must be a finite number in [0, 1)")
    return float(value)
  if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
    raise ValueError(f"{name} must be a positive integer")
  return int(value)


def resolve_model_arguments(args: argparse.Namespace) -> argparse.Namespace:
  """Resolves architecture fields and records the config path on ``args``.

  Reads ``args.config`` when explicitly selected, otherwise ``config.json``
  beside ``args.weights`` unless ``args.no_config`` is true. A missing adjacent
  file permits legacy defaults. An explicitly selected missing file, malformed
  JSON, a non-object config or invalid selected setting raises ValueError or
  OSError before model construction. Unrelated configuration keys are ignored.
  """
  config, path = _read_config(args)
  resolved = {}
  for name, model_key in _ARGUMENT_TO_MODEL_KEY.items():
    explicit = getattr(args, name, None)
    fallback = config.get(name, MODEL_DEFAULTS[model_key])
    value = explicit if explicit is not None else fallback
    resolved[name] = _validate(name, value)
  explicit_variant = getattr(args, "model_variant", None)
  variant = (explicit_variant if explicit_variant is not None
             else config.get("model_variant", DEFAULT_MODEL_VARIANT))
  if not isinstance(variant, str) or variant not in MODEL_VARIANTS:
    raise ValueError(f"model_variant must be one of {MODEL_VARIANTS}")
  resolved["model_variant"] = variant
  for name, value in resolved.items():
    setattr(args, name, value)
  args.training_config_path = path
  return args
