Programming Reference
=============
Welcome to the CNN-Mamba forest typology programming reference page!

This page lists what every module provides, with the tensor shapes each part
expects and returns. `B` is the batch size, `T` the number of seasonal
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

## `mamba_forest.layers`

| Class | Input | Output | Purpose |
|---|---|---|---|
| `Mamba2Block(d_model, d_state, expand, dropout_rate)` | `[B, L, d_model]` | `[B, L, d_model]` | Selective state-space block with a residual connection. |
| `StructureAwareStateFusion()` | `[B, H, W, C]` | `[B, H, W, C]` | Mixes each state with its four direct neighbours. |
| `DynamicMultiModalFusion(d_model, num_modalities, hidden_dim, dropout_rate)` | list of `[B, H, W, C]` | `[B, H, W, d_model]` | Content-dependent attention over the modalities. |

`Mamba2Block` builds three weights of its own: `A_raw` `(d_state,)`, the decay
parameter; `state_to_channel` `(d_state, d_inner)`, the shared projection
between the state space and the channels; and `D` `(d_inner,)`, the per-channel
skip scale on the convolution path. `StructureAwareStateFusion` builds
`neighbour_weights` `(4, C)`, one weight per direction and channel.

`DynamicMultiModalFusion` raises `ValueError` if the number of maps it is given
differs from `num_modalities`; all maps must share `H` and `W`.

## `mamba_forest.model`

| Name | Purpose |
|---|---|
| `OPTICAL_CHANNELS`, `ELEVATION_CHANNELS`, `CLIMATE_CHANNELS` | Channel counts of the three modalities: 10, 3 and 14. |
| `MambaForestSegmenter(num_classes, d_model, dropout_rate, d_state_temporal, d_state_spatial, bottleneck_channels)` | The segmentation network. |

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

`ComboLoss` rejects an `alpha` outside `[0, 1]` and a `class_weights` vector
whose length differs from `num_classes`; passing `class_weights=None` gives
uniform weights. Both `y_true` and `y_pred` are `[B, H, W, num_classes]`, with
`y_true` one-hot and `y_pred` a probability distribution over the last axis.

## `mamba_forest.train`

Command-line entry point, `python -m mamba_forest.train`. Run it with `--help`
for the full list of flags; the defaults reproduce the configuration in
[`results/README.md`](../results/README.md).

| Function | Purpose |
|---|---|
| `build_datasets(args)` | Training and validation datasets from the parsed flags. |
| `cosine_learning_rate(initial, alpha, epoch, total_epochs)` | Cosine decay from `initial` to `alpha * initial`. |
| `scale_loss` / `unscale_gradients` | Dynamic loss scaling under mixed precision, across Keras versions. |
| `f1_from_confusion(confusion)` | Per-class F1 from a confusion matrix, rows = true. |
| `evaluate(model, dataset, num_classes)` | `(pixel accuracy, mean IoU, per-class F1)`, accumulated batch by batch. |
| `forest_scores(per_class_f1)` | `Overall`, `Forests`, `N`, `P` and `TC`. |

Outputs written to `--output-dir`: `config.json` with the resolved flags,
`training_results.csv` with one row per epoch, and the `best.weights.h5` and
`last.weights.h5` checkpoints.

## `mamba_forest.evaluate`

Command-line entry point, `python -m mamba_forest.evaluate --weights ...`. It
rebuilds the network, restores a checkpoint, evaluates the test split and
writes `test_metrics.json` next to the weights. Run it with `--help` for the
full list of flags; the four architecture flags — `--d-model`,
`--d-state-temporal`, `--d-state-spatial` and `--bottleneck-channels` — have to
match the values the checkpoint was trained with, since a checkpoint stores
weights and not the architecture.

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

The same four architecture flags as `evaluate` apply, for the same reason.

## `scripts/plot_results.py`

Plots the training loss and the validation metrics from a
`training_results.csv` produced by `mamba_forest.train`.

```bash
python scripts/plot_results.py runs/mamba_forest/training_results.csv \
    --output runs/mamba_forest/training_curves.png
```
