#!/usr/bin/env python3
"""Local TFRecord checks for split membership and preprocessing contracts."""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mamba_forest import data  # noqa: E402


def _record(label):
  features = {}
  for name, shape in data.FEATURE_SHAPES.items():
    size = int(np.prod(shape))
    if name == "segmentation_labels":
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=[label] * size))
    else:
      feature = tf.train.Feature(
          float_list=tf.train.FloatList(value=np.zeros(size, dtype=np.float32)))
    features[name] = feature
  return tf.train.Example(
      features=tf.train.Features(feature=features)).SerializeToString()


def _write_split(root, split, labels):
  for path, label in zip(data.shard_paths(split, root, len(labels)), labels):
    with tf.io.TFRecordWriter(path) as writer:
      writer.write(_record(label))


def _read_labels(dataset):
  return [int(label) for batch in dataset
          for label in batch["segmentation_labels"][:, 0, 0].numpy()]


def test_shard_selection_is_reproducible_and_changes_by_epoch():
  kwargs = dict(seed=42, shuffle_shards=True, total_shards=32)
  first = data.selected_shard_paths("train", 8, epoch=0, **kwargs)
  assert first == data.selected_shard_paths("train", 8, epoch=0, **kwargs)
  second = data.selected_shard_paths("train", 8, epoch=1, **kwargs)
  assert set(first) != set(second)
  assert len(first) == len(set(first)) == 8
  assert all("-train." in path for path in first + second)


def test_evaluation_selection_preserves_the_leading_shards():
  expected = data.shard_paths("test", total_shards=8)[:3]
  assert data.selected_shard_paths("test", 3, total_shards=8) == expected
  assert data.selected_shard_paths(
      "test", 3, seed=7, epoch=9, total_shards=8) == expected


def test_local_records_keep_splits_separate_and_evaluation_fixed():
  with tempfile.TemporaryDirectory() as root:
    _write_split(root, "train", [1, 2])
    _write_split(root, "validation", [3, 4])
    _write_split(root, "test", [5, 6])
    observed = []
    for split in ("train", "validation", "test"):
      dataset = data.make_dataset(
          split, 2, batch_size=1, training=False, seed=42,
          gcs_root=root, total_shards=2)
      labels = _read_labels(dataset)
      assert labels == _read_labels(dataset)
      observed.append(set(labels))
    assert observed == [{1, 2}, {3, 4}, {5, 6}]
    validation = data.make_dataset(
        "validation", 1, gcs_root=root, total_shards=2)
    assert _read_labels(validation) == [3]
    batch = next(iter(validation))
    assert batch["segmentation_labels"].dtype == tf.int32
    assert np.all(batch["labels"].numpy()[..., 3] == 1)
    # Evaluation applies fixed scaling and no training-time brightness jitter.
    assert np.all(batch["s2"].numpy() == -1)


def test_warmup_iteration_cannot_change_training_shard_membership():
  with tempfile.TemporaryDirectory() as root:
    _write_split(root, "train", [1, 2, 3])
    kwargs = dict(seed=42, epoch=2, gcs_root=root, total_shards=3)
    dataset = data.make_dataset("train", 2, batch_size=1, **kwargs)
    expected_paths = data.selected_shard_paths(
        "train", 2, shuffle_shards=True, **kwargs)
    all_paths = data.shard_paths("train", root, 3)
    expected = [all_paths.index(path) + 1 for path in expected_paths]
    assert _read_labels(dataset.take(1)) == expected[:1]
    assert _read_labels(dataset) == expected
    assert _read_labels(dataset) == expected


def test_parse_preserves_integer_masks():
  example = data.parse_example(_record(2))
  assert example["segmentation_labels"].dtype == tf.int64
  assert example["s2"].shape == data.FEATURE_SHAPES["s2"]
  encoded = data.encode_labels(example)
  assert np.all(encoded["labels"].numpy()[..., 2] == 1)


def test_invalid_labels_are_rejected_before_one_hot():
  @tf.function
  def encode(labels):
    return data.encode_labels({"segmentation_labels": labels})["labels"]

  def encode_eager(labels):
    return data.encode_labels({"segmentation_labels": labels})["labels"]

  for labels in (tf.constant([[-1]], tf.int64),
                 tf.constant([[data.NUM_CLASSES]], tf.int64),
                 tf.constant([[1.5]], tf.float32),
                 tf.constant([[float("nan")]], tf.float32)):
    for encoder in (encode_eager, encode):
      try:
        encoder(labels)
      except tf.errors.InvalidArgumentError:
        continue
      raise AssertionError(f"invalid labels accepted: {labels}")
  valid = encode(tf.constant([[0.0, 8.0]], tf.float32)).numpy()
  assert np.array_equal(valid.argmax(axis=-1), [[0, 8]])


def test_augmentation_keeps_modalities_and_masks_aligned():
  labels = tf.constant([[0, 1, 2], [3, 4, 5]], tf.int64)
  example = {
      "segmentation_labels": labels,
      "s2": tf.cast(labels[None, ..., None], tf.float32),
      "elevation": tf.cast(labels[..., None], tf.float32),
      "climate": tf.ones((4, 14)),
  }
  for seed in range(8):
    changed = data.augment(example, max_brightness_delta=0.0, seed=seed)
    mask = changed["segmentation_labels"].numpy()
    assert np.array_equal(changed["s2"].numpy()[0, ..., 0], mask)
    assert np.array_equal(changed["elevation"].numpy()[..., 0], mask)
    assert np.array_equal(changed["climate"].numpy(), example["climate"].numpy())


def test_selection_and_batch_arguments_are_validated():
  for kwargs in ({"shards_per_epoch": 0}, {"shards_per_epoch": 3},
                 {"shards_per_epoch": 1.5}, {"epoch": -1}, {"epoch": 0.5},
                 {"batch_size": 0}, {"batch_size": 1.5},
                 {"total_shards": 0}, {"shuffle_buffer": -1}):
    arguments = dict(shards_per_epoch=1, total_shards=2, batch_size=1)
    arguments.update(kwargs)
    try:
      data.make_dataset("train", **arguments)
    except ValueError:
      continue
    raise AssertionError(f"invalid dataset argument accepted: {kwargs}")


if __name__ == "__main__":
  tests = [value for name, value in sorted(globals().items())
           if name.startswith("test_") and callable(value)]
  for test in tests:
    test()
    print(f"ok  {test.__name__}")
  print(f"\n{len(tests)} tests passed")
