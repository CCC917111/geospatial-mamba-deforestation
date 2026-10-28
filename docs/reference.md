[Documentation](README.md) · [Project overview](../README.md)

# API reference

Module interfaces, tensor shapes and return values for the package and its
command-line tools. `B` is the batch size, `T` the number of seasonal
composites (4 in ForTy v1), `H` and `W` the tile height and width (128), and
`C` the channel count of the modality in question.

## `mamba_forest.data`

Streaming and preprocessing of the ForTy v1 TFRecord shards.

| Name | Signature | Purpose |
|---|---|---|
| `GCS_ROOT` | `str` | Root of the published shards, `gs://forest_typology/forty_v1/1.0.0`. |
| `TOTAL_SHARDS` | `int` | 1024 shards per split. |
| `NUM_CLASSES` | `int` | 9 land-cover classes. |
| `CLASS_NAMES` | `tuple[str, ...]` | Class names in label order. |
| `FOREST_CLASS_INDICES` | `tuple[int, int, int]` | The three forest classes aggregated into the `Forests` metric. |
| `shard_paths(split, gcs_root, total_shards)` | `-> list[str]` | Full shard list of a split. |
| `selected_shard_paths(split, shards_per_epoch, seed, epoch, ...)` | `-> list[str]` | Explicit shard membership and order for an epoch. |
| `parse_example(proto)` | `-> dict[str, Tensor]` | Parses one serialized example into the documented shapes. |
| `normalize(example)` | `-> dict[str, Tensor]` | Per-modality scaling; climate is z-scored over time. |
| `augment(example, max_brightness_delta, seed)` | `-> dict[str, Tensor]` | Synchronised flips plus optical-only brightness jitter. |
| `encode_labels(example, num_classes)` | `-> dict[str, Tensor]` | Adds the one-hot `labels` tensor. |
| `preprocess(example, num_classes, training)` | `-> dict[str, Tensor]` | Normalisation, optional augmentation, label encoding. |
| `make_dataset(split, shards_per_epoch, ...)` | `-> tf.data.Dataset` | Batched, prefetched dataset over a subset of shards. |
| `split_inputs_labels(batch)` | `-> (dict, Tensor)` | Splits a batch into the model input dict and the one-hot labels. |

A parsed example holds `s2` `[T, H, W, 10]`, `elevation` `[H, W, 3]`,
`climate` `[T, 14]` and `segmentation_labels` `[H, W]`; after `preprocess` it
also holds `labels` `[H, W, 9]`.

`make_dataset` takes `training` and `shuffle_shards`, both defaulting to
`split == "train"`, so evaluation never sees augmented data and always reads
the same shards, which keeps validation metrics comparable across epochs.
Membership stays fixed across iterators. Rebuild the training dataset with a
different zero-based `epoch` to rotate its seeded sample; the training CLI does
this unless `--fixed-train-subset` is set. Invalid class indices are rejected
before one-hot encoding.

## `mamba_forest.layers`

| Class | Input | Output | Purpose |
|---|---|---|---|
| `MambaBlock(d_model, d_state, expand, dropout_rate)` | `[B, L, d_model]` | `[B, L, d_model]` | Mamba-inspired selective state-space block with a residual connection. |
| `StructureAwareStateFusion()` | `[B, H, W, C]` | `[B, H, W, C]` | Fuses each state over its five-point neighbourhood (itself, up, down, left, right). |
| `DynamicMultiModalFusion(d_model, num_modalities, hidden_dim, dropout_rate)` | list of `[B, H, W, C]` | `[B, H, W, d_model]` | Content-dependent attention over the modalities. |

`MambaBlock` builds three weights of its own: `A_raw` `(d_state,)`, the decay
parameter; `state_to_channel` `(d_state, d_inner)`, the shared projection
between the state space and the channels; and `D` `(d_inner,)`, the per-channel
skip scale on the convolution path. `StructureAwareStateFusion` builds
`centre_weight` `(C,)` and `neighbour_weights` `(4, C)`, one weight per position
and channel.

`DynamicMultiModalFusion` raises `ValueError` if the number of maps it is given
differs from `num_modalities`; all maps must share `H` and `W`.

## `mamba_forest.model`

| Name | Purpose |
|---|---|
| `OPTICAL_CHANNELS`, `ELEVATION_CHANNELS`, `CLIMATE_CHANNELS` | Channel counts of the three modalities: 10, 3 and 14. |
| `MambaForestSegmenter(num_classes, d_model, dropout_rate, d_state_temporal, d_state_spatial, bottleneck_channels, model_variant)` | The segmentation network; `cnn_ssm` by default or `cnn_only` for the component ablation. |

`call(inputs, training)` takes a dict with keys `s2` `[B, T, H, W, 10]`,
`elevation` `[B, H, W, 3]` and `climate` `[B, T, 14]`, and returns per-pixel
class probabilities `[B, H, W, num_classes]` at the resolution of the input
tile. `H` and `W` must be divisible by 8, since the encoder pools three times.

Internal helpers: `_encode_optical`, `_encode_elevation` and `_encode_climate`
produce the three `[B, H, W, d_model]` maps; `_encoder_block` and
`_decoder_block` build the convolutional stages.

## `mamba_forest.losses`

| Name | Purpose |
|---|---|
| `DEFAULT_CLASS_WEIGHTS` | Per-class weights of the cross-entropy term. |
| `ComboLoss(num_classes, alpha, label_smoothing, class_weights, smooth)` | `alpha * weighted CE + (1 - alpha) * Dice`. |

`ComboLoss` requires positive integral `num_classes`, `alpha` and
`label_smoothing` in `[0, 1]`, a positive finite `smooth`, and finite
non-negative weights of the correct length. NumPy arrays, lists and tuples
normalize to the same serializable weight tuple; `class_weights=None` gives
uniform weights. Both `y_true` and `y_pred` are `[B, H, W, num_classes]`, with
`y_true` one-hot and `y_pred` a probability distribution over the last axis.

## `mamba_forest.train`

Command-line entry point, `python -m mamba_forest.train`. Run it with `--help`
for the full list of flags; the defaults reproduce the configuration in
[`results/README.md`](../results/README.md).

| Function | Purpose |
|---|---|
| `configure_precision(precision, no_mixed_precision)` | Resolves auto from visible GPUs and explicitly sets the compute policy. |
| `DatasetEvaluator(model, num_classes)` | Callable evaluator retaining one graph while resetting confusion for each pass; keep it separate from model checkpoint state. |
| `prepare_output_directory(output_dir)` | Creates a run directory; rejects existing checkpoint weights before config writes. |
| `build_datasets(args)` | Training and validation datasets from the parsed flags. |
| `build_train_dataset(args, epoch)` | Explicit training subset for a zero-based epoch. |
| `training_epoch_manifest(args, epoch)` | Paths and status for a one-based training epoch. |
| `write_split_manifest(path, manifest)` | Atomic manifest write with completed shard coverage. |
| `cosine_learning_rate(initial, alpha, epoch, total_epochs)` | Cosine decay from `initial` to `alpha * initial`. |
| `scale_loss` / `unscale_gradients` | Dynamic loss scaling under mixed precision, across Keras versions. |
| `f1_from_confusion(confusion)` | Per-class F1 from a confusion matrix, rows = true. |
| `evaluate(model, dataset, num_classes, return_confusion=False)` | Graph-accumulated `(accuracy, mean IoU, F1)`; optional fourth confusion matrix. Empty input is an error. |
| `forest_scores(per_class_f1)` | `Overall`, `Forests`, `N`, `P` and `TC`. |

Outputs written to `--output-dir`: `config.json` with the resolved flags,
`training_results.csv` with metrics and elapsed training/validation times,
`split_manifest.json` with exact shard paths, and the `best.weights.h5` and
`last.weights.h5` checkpoints.
An optional `--selection-seed` separates shard membership from run randomness.
New histories record training batches, optimizer updates and skipped updates;
zero-progress epochs fail. Histories and H5 files use temporary siblings and
filesystem replacement to preserve previous complete artifacts on save failure.

## `mamba_forest.evaluate`

Command-line entry point, `python -m mamba_forest.evaluate --weights ...`. It
rebuilds the network, restores a checkpoint, evaluates the test split and
writes `test_metrics.json` next to the weights, including the resolved model
config and training-config path when used. Architecture flags override the
training `config.json` beside the checkpoint; unspecified values come from
that file or the original model defaults. The resolved values must match the
weights. `--config` selects a different local training-config path and
`--no-config` disables discovery. Neither option inherits data paths or
runtime precision.

## `mamba_forest.checkpoint`

| Name | Purpose |
|---|---|
| `add_model_arguments(parser)` | Adds architecture flags and mutually exclusive `--config` / `--no-config`. |
| `resolve_model_arguments(args)` | Resolves CLI, saved config and defaults in that order; validates effective values and returns the updated namespace. |
| `save_weights_atomic(model, path)` | Saves a nonempty sibling H5, hashes it and replaces the destination only after success; returns the SHA-256. Weights only, not optimizer resumption. |

The resolver preserves the legacy `args.dropout` name and records
`args.training_config_path`. It only reads the five recognized architecture
settings from a flat training JSON object. A missing implicit sidecar uses
defaults; explicit missing files, malformed JSON and invalid effective values
raise an error. Positive integer capacities and finite dropout in `[0, 1)`
are required. `model.MODEL_DEFAULTS` is the shared source of model defaults.

## Retained research modules

- `python -m mamba_forest.stage_data`: stages only a fixed cohort with publisher
  filenames, size/SHA receipts, immutable cache repair and byte/disk limits.
- `python -m mamba_forest.research`: freezes a three-seed architecture pilot,
  continues verified completed stages, preserves partial attempts and gates
  test evaluation on complete training. Planning imports no TensorFlow.
- `python -m mamba_forest.research_report`: validates complete checkpoint/data
  evidence and writes numeric-ID metrics plus a retention manifest. Real-pilot
  completion requires the official source and source/runtime/GPU/data receipts.

See the [research guide](research-pilot.md) for protocols, continuation and
limitations. Existing `experiment` executions remain fresh-root runs;
`research` provides the separate continuation workflow.

## `mamba_forest.predict`

Command-line entry point, `python -m mamba_forest.predict --weights ... \
--input-dir ... --output-dir ...`. It applies a checkpoint to a directory of
`.npz` tiles and writes a class map per tile plus a `predictions.csv` of class
shares. [Mapping Your Own Region](apply.md) documents the input contract and
the outputs.

| Name | Purpose |
|---|---|
| `REQUIRED_ARRAYS` | The three arrays a tile file must contain: `s2`, `elevation`, `climate`. |
| `CLASS_COLORS` | One RGB triple per class, in the order of `CLASS_NAMES`. |
| `load_tile(path)` | Reads a tile and validates its shapes against the contract. |
| `preprocess_tile(tile)` | Applies `data.normalize` and adds the batch axis. |
| `colourise(class_map)` | `[H, W]` class indices to an `[H, W, 3]` `uint8` image. |
| `class_shares(class_map, num_classes)` | Pixel fraction per class. |

The same architecture/config resolution as `evaluate` applies.

`--timings PATH` saves stage measurements from the normal prediction path.
The report separates first-inference tracing from later predictions and
records the checkpoint, input dimensions and selected export options.
`--batch-size N` defaults to one and groups consecutive equal-shape tiles up
to that bound after per-tile normalization. Shape changes flush partial
batches. Version-two reports retain actual timings in `batches`; tile sample
costs allocate shared assembly/inference time equally and are labeled as
amortized costs. The entire first batch is marked as first inference.

## `mamba_forest.benchmark_predict`

Repeats the production prediction path with full-input warmup passes and
measured passes. It retains raw timing samples and distinct output directories
while loading the checkpoint once. Process startup is excluded; runtime import
cost is recorded separately. See [Local demo and timing](local-demo.md).
The same `--batch-size` control applies. Full-pass tiles/s includes input and
output handling. Separate `steady_state_batch_stages` and tile-stage summaries
keep observed batch latency distinct from allocated per-tile costs.

## `mamba_forest.demo`

Generates synthetic 128 × 128 ForTy-format records and runs the real training,
evaluation and prediction commands on CPU. It writes a comparison figure and
a run summary in a fresh output directory. See [Local demo](local-demo.md).

## `scripts/plot_results.py`

Plots the training loss and the validation metrics from a
`training_results.csv` produced by `mamba_forest.train`.

```bash
python scripts/plot_results.py runs/mamba_forest/training_results.csv \
    --output runs/mamba_forest/training_curves.png
```
