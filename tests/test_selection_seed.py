"""Independent shard cohorts preserve run randomness and legacy defaults."""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np
import pytest
import tensorflow as tf

from mamba_forest import data, experiment, train, validate_data


def arguments(seed=42, selection_seed=None):
  return argparse.Namespace(
      seed=seed, selection_seed=selection_seed, train_shards=3, val_shards=1,
      test_shards=1, batch_size=1, fixed_train_subset=True, gcs_root="records",
      total_shards=8, require_gpu=False)


def test_fixed_cohort_across_model_seeds_matches_preflight(monkeypatch):
  monkeypatch.setattr(tf.config, "list_physical_devices", lambda: [])
  monkeypatch.setattr(validate_data, "inspect_record",
                      lambda path, result: result.update(metadata={}))
  cohorts = []
  for seed in (42, 43, 44):
    args = arguments(seed, 2026)
    membership = train.training_epoch_manifest(args, 1)["paths"]
    cohorts.append(membership)
    assert train.training_epoch_manifest(args, 10)["paths"] == membership
    preflight = validate_data.run_preflight(args)
    assert preflight["status"] == "passed"
    assert preflight["splits"]["train"]["selected_shard_paths"] == membership
  assert cohorts[0] == cohorts[1] == cohorts[2]
  assert train.training_epoch_manifest(arguments(42), 1)["paths"] != (
      train.training_epoch_manifest(arguments(43), 1)["paths"])


def test_dataset_cohort_is_independent_of_example_shuffle_seed(tmp_path, monkeypatch):
  shapes = {"s2": (4, 8, 8, 10), "elevation": (8, 8, 3),
            "climate": (4, 14), "segmentation_labels": (8, 8)}
  monkeypatch.setattr(data, "FEATURE_SHAPES", shapes)
  monkeypatch.setattr(data, "_FEATURE_DESCRIPTION", {
      key: tf.io.FixedLenFeature([int(np.prod(shape))],
                                 tf.int64 if key == "segmentation_labels"
                                 else tf.float32)
      for key, shape in shapes.items()})
  for index, path in enumerate(data.shard_paths("train", str(tmp_path), 8)):
    features = {}
    for key, shape in shapes.items():
      if key == "segmentation_labels":
        value = tf.train.Feature(int64_list=tf.train.Int64List(
            value=[index] * int(np.prod(shape))))
      else:
        value = tf.train.Feature(float_list=tf.train.FloatList(
            value=[0.0] * int(np.prod(shape))))
      features[key] = value
    example = tf.train.Example(features=tf.train.Features(feature=features))
    with tf.io.TFRecordWriter(path) as writer:
      writer.write(example.SerializeToString())
  shuffle_seeds = []
  original_shuffle = tf.data.Dataset.shuffle

  def shuffle(dataset, buffer_size, seed=None, **kwargs):
    shuffle_seeds.append(seed)
    return original_shuffle(dataset, buffer_size, seed=seed, **kwargs)

  monkeypatch.setattr(tf.data.Dataset, "shuffle", shuffle)
  memberships = []
  for seed in (42, 43):
    dataset = data.make_dataset(
        "train", 3, batch_size=1, training=False, shuffle_shards=True,
        shuffle_buffer=3, seed=seed, selection_seed=2026, total_shards=8,
        gcs_root=str(tmp_path))
    memberships.append(sorted(int(batch["segmentation_labels"][0, 0, 0])
                              for batch in dataset))
  assert memberships[0] == memberships[1]
  assert shuffle_seeds == [42, 43]


def test_training_passes_model_and_membership_seeds_separately(monkeypatch):
  calls = []
  monkeypatch.setattr(data, "make_dataset",
                      lambda *args, **kwargs: calls.append(kwargs))
  train.build_train_dataset(arguments(43, 2026), 5)
  assert calls[0]["seed"] == 43
  assert calls[0]["selection_seed"] == 2026
  assert calls[0]["epoch"] == 0


def test_plan_propagates_shared_selection_seed_to_every_training_preflight(tmp_path):
  args = experiment.parse_args([
      "--run-root", str(tmp_path), "--selection-seed", "2026",
      "--seeds", "42", "43", "44", "--comparison", "architecture"])
  plan = experiment.make_plan(args)
  assert plan["config"]["selection_seed"] == 2026
  for step in plan["steps"]:
    if step["phase"] not in ("train", "preflight"):
      continue
    argv = step["argv"]
    assert argv[argv.index("--selection-seed") + 1] == "2026"
    assert int(argv[argv.index("--seed") + 1]) in (42, 43, 44)


@pytest.mark.parametrize("module", [train, validate_data, experiment])
@pytest.mark.parametrize("value", [-1, 2**32])
def test_invalid_selection_seed_rejected_by_cli(module, value, tmp_path, monkeypatch):
  flags = ["program", "--selection-seed", str(value)]
  if module is experiment:
    flags += ["--run-root", str(tmp_path / "unused")]
  monkeypatch.setattr(sys, "argv", flags)
  with pytest.raises(SystemExit):
    module.parse_args()
  assert not (tmp_path / "unused").exists()


def test_optimizer_counts_expose_skipped_updates_and_reject_no_progress():
  assert train.optimizer_epoch_counts(2, 5, 4) == {
      "train_batches": 4, "optimizer_updates": 3, "skipped_optimizer_updates": 1}
  with pytest.raises(RuntimeError, match="no optimizer updates"):
    train.optimizer_epoch_counts(2, 2, 4)
  with pytest.raises(RuntimeError, match="batch budget"):
    train.optimizer_epoch_counts(2, 7, 4)


def test_atomic_history_retains_previous_complete_csv_on_failure(tmp_path, monkeypatch):
  path = tmp_path / "training_results.csv"
  train.write_training_history(path, ["epoch", "score"], [[1, 0.12345671]])
  previous = path.read_bytes()
  original_replace = train.os.replace

  def failed_replace(*args):
    raise OSError("simulated failed replacement")

  monkeypatch.setattr(train.os, "replace", failed_replace)
  with pytest.raises(OSError, match="simulated"):
    train.write_training_history(path, ["epoch", "score"], [[1, 0.12345674]])
  assert path.read_bytes() == previous
  assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]
  monkeypatch.setattr(train.os, "replace", original_replace)
  train.write_training_history(path, ["epoch", "score"], [[1, 0.12345674]])
  with path.open() as handle:
    row = next(csv.DictReader(handle))
  assert float(row["score"]) == 0.12345674
  assert row["epoch"] == "1"


def test_legacy_gradient_finiteness_counts_dense_and_sparse_skips():
  finite = tf.constant([1.0, 2.0])
  assert bool(train.gradients_are_finite([finite, None]))
  sparse = tf.IndexedSlices(tf.constant([float("inf")]), tf.constant([0]))
  assert not bool(train.gradients_are_finite([finite, sparse]))
  with pytest.raises(ValueError, match="no gradients"):
    train.gradients_are_finite([None])


def test_actual_loss_scaler_skips_are_visible_in_epoch_counts():
  variable = tf.Variable(1.0)
  optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
      tf.keras.optimizers.SGD(learning_rate=0.1), initial_scale=8.0)
  if train._uses_keras2_loss_scaling(optimizer):
    pytest.skip("Legacy scalers use the separately tested finite-gradient counter")
  start = int(optimizer.iterations.numpy())
  optimizer.apply_gradients([(tf.constant(float("inf")), variable)])
  optimizer.apply_gradients([(train.scale_loss(optimizer, tf.constant(1.0)), variable)])
  assert train.optimizer_epoch_counts(start, int(optimizer.iterations.numpy()), 2) == {
      "train_batches": 2, "optimizer_updates": 1, "skipped_optimizer_updates": 1}
  assert float(variable.numpy()) == pytest.approx(0.9)
