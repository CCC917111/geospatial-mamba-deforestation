"""Input pipeline for the ForTy v1 forest-typology dataset.

ForTy v1 is published as 1024 TFRecord shards per split under
``gs://forest_typology/forty_v1/1.0.0/``. Reading a sample of shards per epoch
keeps experiments on a single Colab GPU tractable: one shard holds a few
hundred 128x128 tiles, so 32 shards is roughly 2,500 tiles per epoch.

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

# Class names as documented with the dataset. NOTE: the index of the first
# forest class has not been verified against the dataset metadata; see
# FOREST_CLASS_INDICES below and the "Known limitations" section of the README.
CLASS_NAMES = (
    "Natural Forest",
    "Planted Forest",
    "Tree Crops",
    "Other Vegetation",
    "Water",
    "Ice",
    "Bare Ground",
    "Built Areas",
    "Unknown",
)

# Classes aggregated into the "forest" metric reported during training.
FOREST_CLASS_INDICES = (0, 1, 2)

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


def preprocess(example, num_classes: int = NUM_CLASSES):
  """Normalises the modalities and one-hot encodes the labels.

  The scaling constants are fixed rather than dataset statistics: Sentinel-2
  reflectance is divided by 5000, elevation/slope/aspect by their physical
  ranges, and each climate variable is standardised over the time axis of the
  example.
  """
  processed = {}
  processed["s2"] = tf.cast(example["s2"], tf.float32) / 5000.0 - 1.0

  elevation = example["elevation"]
  processed["elevation"] = tf.stack([
      (elevation[..., 0] - 2000.0) / 3000.0,
      elevation[..., 1] / 45.0 - 1.0,
      elevation[..., 2] / 180.0 - 1.0,
  ], axis=-1)

  # Standardise each climate variable over time; pooling the 14 physically
  # different variables into one mean and variance would mostly measure the
  # spread between variables (e.g. temperature vs. precipitation).
  climate = example["climate"]
  mean, variance = tf.nn.moments(climate, axes=[0], keepdims=True)
  processed["climate"] = (climate - mean) * tf.math.rsqrt(variance + 1e-6)

  labels = tf.cast(example["segmentation_labels"], tf.int32)
  processed["segmentation_labels"] = labels
  processed["labels"] = tf.one_hot(labels, num_classes)
  return processed


def make_dataset(split: str, shards_per_epoch: int, batch_size: int = 8,
                 shuffle_shards: bool | None = None, shuffle_buffer: int = 0,
                 seed: int | None = None, gcs_root: str = GCS_ROOT,
                 total_shards: int = TOTAL_SHARDS,
                 num_classes: int = NUM_CLASSES) -> tf.data.Dataset:
  """Builds a batched dataset from a subset of the split's shards.

  Args:
    split: "train", "validation" or "test".
    shards_per_epoch: how many shards to read. Use ``total_shards`` for all.
    batch_size: examples per batch.
    shuffle_shards: whether to pick a random subset of shards. Defaults to True
      for the training split and False otherwise, so that evaluation always
      sees the same examples and metrics are comparable across epochs.
    shuffle_buffer: if > 0, shuffle examples with this buffer size.
    seed: seed for the shard and example shuffling.
    gcs_root: root of the TFRecord shards.
    total_shards: number of shards per split.
    num_classes: number of segmentation classes.
  """
  if shards_per_epoch < 1 or shards_per_epoch > total_shards:
    raise ValueError(
        f"shards_per_epoch must be in [1, {total_shards}], got {shards_per_epoch}")
  if shuffle_shards is None:
    shuffle_shards = split == "train"

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
      lambda ex: preprocess(ex, num_classes), num_parallel_calls=tf.data.AUTOTUNE)
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
