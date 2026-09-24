Common Tasks
=============
Welcome to the CNN-Mamba forest typology common tasks page!

This page gives the recipes for the changes people most often want to make.

## Add a fourth modality

ForTy v1 also ships Sentinel-1 radar, ascending and descending. To bring one
in:

1. Add its key and shape to `FEATURE_SHAPES` and `_FEATURE_DESCRIPTION` in
   `src/mamba_forest/data.py`, so the parser reads it out of the TFRecord.
2. Give it a scaling rule in `normalize`. Radar backscatter is in decibels, so
   `(x + 15) / 15` puts the usual range near `[-1, 1]`.
3. Add its key to the geometric part of `augment` if it is a spatial modality,
   and leave it out of the brightness jitter, which is optical-only by design.
4. Add it to the input dict built by `split_inputs_labels`.
5. In `src/mamba_forest/model.py`, give it an encoder — a second temporal Mamba
   block if it is a time series, a convolutional stack if it is a static map —
   and raise `num_modalities` in the `DynamicMultiModalFusion` constructor to
   4. The fusion layer needs no other change: it pools whatever it is given and
   emits one softmax weight per modality.
6. Add a case to `tests/test_model.py::test_every_modality_influences_the_prediction`.

## Change the capacity of the network

`MambaForestSegmenter` takes `d_model`, `d_state_temporal`, `d_state_spatial`
and `bottleneck_channels`. `d_model` sets the width after fusion and therefore
the width of the whole U-Net; `--d-model` on the training entry point is the
flag to reach for first. The encoder and decoder channel schedule lives in the
`_encoder_block` and `_decoder_block` calls in `__init__` and is deliberately
explicit, so changing `64 / 128 / 256` is a one-line edit per stage.

Whatever you change, the same value has to be passed to
`python -m mamba_forest.evaluate --d-model`, since the checkpoint only stores
weights.

## Change the objective

`ComboLoss` takes `alpha`, `label_smoothing` and `class_weights`.

- `--combo-alpha 1.0` gives a pure weighted cross-entropy,
  `--combo-alpha 0.0` a pure Dice loss.
- `--no-class-weights` switches to uniform weights, which is the ablation
  behind the paragraph on class collapse in [`results`](../results/README.md).
- A different weight vector goes into `DEFAULT_CLASS_WEIGHTS` in
  `src/mamba_forest/losses.py`; its length must equal `NUM_CLASSES`.

To add a term of your own, implement it as a method on `ComboLoss` next to
`_dice` and combine it in `call`; keeping it inside one loss class means the
training loop does not change.

## Run on a different dataset

The network only assumes an optical time series, a static spatial map and a
low-dimensional series, so a different multi-modal benchmark mostly means a new
data module:

1. Copy `data.py` and adapt `FEATURE_SHAPES`, `_FEATURE_DESCRIPTION`,
   `shard_paths` and `normalize` to the new source.
2. Set `NUM_CLASSES`, `CLASS_NAMES` and `FOREST_CLASS_INDICES` — the last one
   is what `forest_scores` aggregates, so it can point at whatever group of
   classes the new benchmark reports.
3. Adjust the channel constants at the top of `model.py`.
4. Update `CLASS_COLORS` in `predict.py` so the colour map has one entry per
   class.

Tile size is free as long as it is divisible by 8, since the encoder pools
three times; the network is fully convolutional and returns a map at the
resolution it was given.

## Train on the full dataset

`--train-shards 1024` reads the whole training split in one epoch; the default
of 128 is the reduced budget the reported numbers were produced under. Expect
to raise `--epochs` and to lower `--patience` accordingly, and keep
`--val-shards` fixed so the validation metrics stay comparable.

## Work on the JAX track

`jeo_plugin/` is independent of the TensorFlow package and is used by copying
its three files into a JEO checkout; see
[`jeo_plugin/README.md`](../jeo_plugin/README.md). Its tests run without a JEO
checkout, so the model can be developed on its own:

```bash
pip install jax flax einops
python jeo_plugin/tests/test_mamba_mtst_shapes.py
```
