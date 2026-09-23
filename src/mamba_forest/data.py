"""Input pipeline for the ForTy v1 forest-typology dataset.

ForTy v1 is published as 1024 TFRecord shards per split under
``gs://forest_typology/forty_v1/1.0.0/``. Reading a sample of shards per epoch
keeps experiments on a single GPU tractable: one shard holds a few hundred
128x128 tiles, so the default of 128 training shards is roughly 12.5% of the
training split per epoch.

Each example used here contains:

  s2                   (4, 128, 128, 10)  Sentinel-2, 4 seasonal composites
  elevation            (128, 128, 3)      elevation, slope, aspect
  climate              (4, 14)            14 climate variables per season
  segmentation_labels  (128, 128)         per-pixel class index
"""

from __future__ import annotations

import tensorflow as tf

GCS_ROOT = "gs://forest_typology/forty_v1/1.0.0"
TOTAL_SHARDS = 1024

NUM_CLASSES = 9

# Class order of the ForTy v1 segmentation labels.
CLASS_NAMES = (
    "Unknown",
    "Natural Forest",
    "Planted Forest",
    "Tree Crops",
    "Other Vegetation",
    "Water",
    "Ice",
    "Bare Ground",
    "Built Areas",
)

# The three forest types the benchmark aggregates into its "Forests" metric.
FOREST_CLASS_INDICES = (1, 2, 3)
NATURAL_FOREST_INDEX = 1
PLANTED_FOREST_INDEX = 2
TREE_CROPS_INDEX = 3

FEATURE_SHAPES = {
    "s2": (4, 128, 128, 10),
    "elevation": (128, 128, 3),
    "climate": (4, 14),
    "segmentation_labels": (128, 128),
}

_FEATURE_DESCRIPTION = {
    "s2": tf.io.FixedLenFeature([4 * 128 * 128 * 10], tf.float32),
    "elevation": tf.io.FixedLenFeature([128 * 128 * 3], tf.float32),
    "climate": tf.io.FixedLenFeature([4 * 14], tf.float32),
    "segmentation_labels": tf.io.FixedLenFeature([128 * 128], tf.int64),
}


def shard_paths(split: str, gcs_root: str = GCS_ROOT,
                total_shards: int = TOTAL_SHARDS) -> list[str]:
  """Returns the full list of shard paths of a split."""
  if split not in ("train", "validation", "test"):
    raise ValueError(f"Unknown split: {split}")
  return [
      f"{gcs_root}/forty_v1-{split}.tfrecord-{i:05d}-of-{total_shards:05d}"
      for i in range(total_shards)
  ]


def parse_example(example_proto):
  """Parses one serialized example into tensors of the documented shapes."""
  parsed = tf.io.parse_single_example(example_proto, _FEATURE_DESCRIPTION)
  example = {}
  for key, shape in FEATURE_SHAPES.items():
    value = tf.reshape(parsed[key], shape)
    example[key] = tf.cast(value, tf.float32) if value.dtype == tf.int64 else value
  return example


def normalize(example):
  """Scales the modalities with fixed, physically motivated constants.

  Sentinel-2 reflectance is mapped to [-1, 1] with a factor of 5000; elevation,
  slope and aspect are centred and scaled by their physical ranges; and each
  climate variable is standardised over the time axis of the example, so the
  network sees the seasonal anomaly rather than the absolute level.
  """
  processed = dict(example)
  processed["s2"] = tf.cast(example["s2"], tf.float32) / 5000.0 - 1.0

  elevation = tf.cast(example["elevation"], tf.float32)
  processed["elevation"] = tf.stack([
      (elevation[..., 0] - 2000.0) / 3000.0,
      elevation[..., 1] / 45.0 - 1.0,
      elevation[..., 2] / 180.0 - 1.0,
  ], axis=-1)

  # Standardise each climate variable over time. Pooling the 14 physically
  # different variables into one mean and variance would mostly measure the
  # spread between variables, e.g. temperature against precipitation.
  climate = tf.cast(example["climate"], tf.float32)
  mean, variance = tf.nn.moments(climate, axes=[0], keepdims=True)
  processed["climate"] = (climate - mean) * tf.math.rsqrt(variance + 1e-6)
  return processed


def augment(example, max_brightness_delta: float = 0.1, seed: int | None = None):
  """Applies the training-time augmentation of the paper.

  Satellite imagery is acquired from nadir, so the semantic class of a region
  is invariant to mirroring: horizontal and vertical flips are each applied
  with probability 0.5. Both flips are drawn once per example and applied to
  the optical series, the elevation map and the label mask together, so the
  pixel-wise correspondence required for segmentation is preserved. The
  brightness jitter simulates illumination and atmospheric variation and is
  applied to the optical modality only; perturbing elevation or climate would
  change the physical meaning of those measurements rather than model sensor
  noise.
  """
  # Two independent draws: the same op-level seed would make both ops return
  # the same value, so the tile could only ever be left untouched or rotated
  # by 180 degrees.
  flip_lr = tf.random.uniform([], seed=seed) > 0.5
  flip_ud = tf.random.uniform([], seed=None if seed is None else seed + 1) > 0.5

  def apply_flips(tensor):
    # Works for [T, H, W, C], [H, W, C] and [H, W]: the flips act on the two
    # axes in front of the (optional) channel axis.
    rank = len(tensor.shape)
    if rank == 2:
      x = tensor[..., None]
    else:
      x = tensor
    x = tf.cond(flip_lr, lambda: tf.image.flip_left_right(x), lambda: x)
    x = tf.cond(flip_ud, lambda: tf.image.flip_up_down(x), lambda: x)
    if rank == 2:
      x = tf.squeeze(x, axis=-1)
    return x

  processed = dict(example)
  for key in ("s2", "elevation", "segmentation_labels"):
    if key in example:
      processed[key] = apply_flips(example[key])

  delta = tf.random.uniform(
      [], -max_brightness_delta, max_brightness_delta,
      seed=None if seed is None else seed + 2)
  processed["s2"] = processed["s2"] + delta
  return processed


def encode_labels(example, num_classes: int = NUM_CLASSES):
  """Adds the one-hot label tensor required by the categorical losses."""
  processed = dict(example)
  labels = tf.cast(example["segmentation_labels"], tf.int32)
  processed["segmentation_labels"] = labels
  processed["labels"] = tf.one_hot(labels, num_classes)
  return processed


def preprocess(example, num_classes: int = NUM_CLASSES, training: bool = False):
  """Normalisation, optional augmentation and one-hot label encoding.

  The augmentation is intentionally left unseeded here: a fixed op-level seed
  inside a ``tf.data`` map would draw the same flips for every example.
  ``augment`` takes a seed of its own for use in tests.
  """
  processed = normalize(example)
  if training:
    processed = augment(processed)
  return encode_labels(processed, num_classes)


def make_dataset(split: str, shards_per_epoch: int, batch_size: int = 16,
                 training: bool | None = None, shuffle_shards: bool | None = None,
                 shuffle_buffer: int = 0, seed: int | None = None,
                 gcs_root: str = GCS_ROOT, total_shards: int = TOTAL_SHARDS,
                 num_classes: int = NUM_CLASSES) -> tf.data.Dataset:
  """Builds a batched dataset from a subset of the split's shards.

  Args:
    split: "train", "validation" or "test".
    shards_per_epoch: how many shards to read. Use ``total_shards`` for all.
    batch_size: examples per batch.
    training: whether to apply augmentation. Defaults to True for the training
      split and False otherwise, so that evaluation never sees augmented data.
    shuffle_shards: whether to pick a random subset of shards. Defaults to the
      value of ``training``, so that evaluation always sees the same examples
      and metrics are comparable across epochs.
    shuffle_buffer: if > 0, shuffle examples with this buffer size.
    seed: seed for the shard and example shuffling, so a run reads the shards
      in a reproducible order.
    gcs_root: root of the TFRecord shards.
    total_shards: number of shards per split.
    num_classes: number of segmentation classes.
  """
  if shards_per_epoch < 1 or shards_per_epoch > total_shards:
    raise ValueError(
        f"shards_per_epoch must be in [1, {total_shards}], got {shards_per_epoch}")
  if training is None:
    training = split == "train"
  if shuffle_shards is None:
    shuffle_shards = training

  paths = shard_paths(split, gcs_root, total_shards)
  files = tf.data.Dataset.from_tensor_slices(paths)
  if shuffle_shards:
    # Reshuffling every epoch lets long runs cover more of the dataset.
    files = files.shuffle(total_shards, seed=seed, reshuffle_each_iteration=True)
  files = files.take(shards_per_epoch)

  dataset = files.interleave(
      tf.data.TFRecordDataset,
      cycle_length=min(8, shards_per_epoch),
      num_parallel_calls=tf.data.AUTOTUNE,
      deterministic=not shuffle_shards)
  dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
  dataset = dataset.map(
      lambda ex: preprocess(ex, num_classes, training),
      num_parallel_calls=tf.data.AUTOTUNE)
  if shuffle_buffer > 0:
    dataset = dataset.shuffle(shuffle_buffer, seed=seed)
  return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def split_inputs_labels(batch):
  """Splits a batch into the model input dict and the one-hot labels."""
  inputs = {
      "s2": batch["s2"],
      "elevation": batch["elevation"],
      "climate": batch["climate"],
  }
  return inputs, batch["labels"]
