"""Preprocessing ops for ForTy v1, registered with the JEO pp builder.

Copy this file to ``jeo/pp/forty_v1_ops.py`` in a JEO checkout (see
``jeo_plugin/README.md``); the ops are then available in the ``pp_train`` /
``pp_eval`` strings of a config.

Ops (applied in this order):

  forty_v1_normalize      scale the raw modalities to roughly [-1, 1]
  forty_v1_mask_invalid   replace invalid pixels using the per-modality masks
  forty_v1_augment        random flips and 90-degree rotations (training only)
  forty_v1_to_mamba       rename the keys to the model's modality names
"""

from __future__ import annotations

import tensorflow as tf
from jeo.pp import pp_builder

SPATIAL_KEYS = ("s2", "s1_asc", "s1_desc", "elevation", "segmentation_labels")


@pp_builder.Registry.register("preprocess_ops.forty_v1_normalize")
def forty_v1_normalize(**kwargs):
  """Scales the modalities with fixed, physically motivated constants."""
  del kwargs

  def _normalize(data):
    result = dict(data)
    if "s2" in data:
      result["s2"] = tf.cast(data["s2"], tf.float32) / 5000.0 - 1.0
    for key in ("s1_asc", "s1_desc"):
      if key in data:
        result[key] = (tf.cast(data[key], tf.float32) + 15.0) / 15.0
    if "elevation" in data:
      elevation = tf.cast(data["elevation"], tf.float32)
      result["elevation"] = tf.stack([
          (elevation[..., 0] - 2000.0) / 3000.0,
          elevation[..., 1] / 45.0 - 1.0,
          elevation[..., 2] / 180.0 - 1.0,
      ], axis=-1)
    if "climate" in data:
      climate = tf.cast(data["climate"], tf.float32)
      # Standardise over every axis but the last one. Reducing over the
      # singleton spatial axes as well would make the mean equal to the input
      # and zero the whole modality.
      axes = list(range(climate.shape.rank - 1))
      mean, variance = tf.nn.moments(climate, axes=axes, keepdims=True)
      result["climate"] = (climate - mean) * tf.math.rsqrt(variance + 1e-6)
    return result

  return _normalize


@pp_builder.Registry.register("preprocess_ops.forty_v1_mask_invalid")
def forty_v1_mask_invalid(fill_value: float = 0.0, **kwargs):
  """Sets pixels whose mask is zero to ``fill_value``."""
  del kwargs

  def _mask_invalid(data):
    result = dict(data)
    for key in ("s2", "s1_asc", "s1_desc"):
      mask_key = f"{key}_mask"
      if key in data and mask_key in data:
        valid = tf.cast(data[mask_key] > 0, tf.float32)
        result[key] = data[key] * valid + fill_value * (1.0 - valid)
    return result

  return _mask_invalid


@pp_builder.Registry.register("preprocess_ops.forty_v1_augment")
def forty_v1_augment(**kwargs):
  """Applies the same random flips and rotation to every spatial modality."""
  del kwargs

  def _augment_one(tensor, flip_lr, flip_ud, rot_k):
    """Handles [H, W], [H, W, C] and [T, H, W, C] tensors."""
    rank = tensor.shape.rank
    if rank == 2:
      x = tensor[..., None]
    else:
      x = tensor

    x = tf.cond(flip_lr, lambda: tf.image.flip_left_right(x), lambda: x)
    x = tf.cond(flip_ud, lambda: tf.image.flip_up_down(x), lambda: x)
    x = tf.image.rot90(x, k=rot_k)

    if rank == 2:
      x = tf.squeeze(x, axis=-1)
    return x

  def _augment(data):
    # One decision per example, shared by all modalities so that the pixels
    # stay aligned with the labels.
    flip_lr = tf.random.uniform([]) > 0.5
    flip_ud = tf.random.uniform([]) > 0.5
    rot_k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)

    result = dict(data)
    for key in SPATIAL_KEYS:
      if key in data:
        result[key] = _augment_one(data[key], flip_lr, flip_ud, rot_k)
    return result

  return _augment


@pp_builder.Registry.register("preprocess_ops.forty_v1_to_mamba")
def forty_v1_to_mamba(keep_metadata: bool = True, **kwargs):
  """Renames the dataset keys to the modality names used by the model."""
  del kwargs
  rename = {
      "s2": "optical",
      "s1_asc": "sar_asc",
      "s1_desc": "sar_desc",
      "climate": "climate",
  }

  def _to_mamba(data):
    result = {new: data[old] for old, new in rename.items() if old in data}
    if "elevation" in data:
      # Add a length-1 time axis so that the model sees [T, H, W, C].
      result["elevation"] = tf.expand_dims(data["elevation"], axis=0)
    if "segmentation_labels" in data:
      result["labels"] = data["segmentation_labels"]
    if keep_metadata:
      for key in ("id", "lat", "lon"):
        if key in data:
          result[key] = data[key]
    return result

  return _to_mamba
