Developer page
=============
Welcome to the CNN-Mamba forest typology developer main page!

This page is the entry point for working on the code: how the repository is
laid out, how a batch travels through it, and where to start when you want to
change something.

## Abstract

The project implements an efficient multi-modal CNN-Mamba framework for
pixel-level forest type segmentation on the ForTy v1 benchmark. Convolutional
encoders extract local spatial structure, selective state-space (Mamba) blocks
model seasonal dynamics and global context in linear time, a dynamic MLP-based
fusion mechanism weighs the modalities per sample, and a U-Net-style
encoder-decoder with a Mamba bottleneck reconstructs the segmentation map.

There are two independent implementations: a TensorFlow package under
`src/mamba_forest/`, which is the one the reported numbers come from, and a
JAX/Flax plug-in for DeepMind's JEO framework under `jeo_plugin/`.

## Layout

```
src/mamba_forest/
  data.py        ForTy v1 streaming, normalisation, augmentation, batching
  layers.py      Mamba2Block, StructureAwareStateFusion, DynamicMultiModalFusion
  model.py       MambaForestSegmenter, the five-stage network
  losses.py      ComboLoss, weighted cross-entropy plus soft Dice
  train.py       training loop, metrics, checkpointing, CLI
  evaluate.py    test-split evaluation, CLI
  predict.py     apply a checkpoint to your own tiles, CLI
jeo_plugin/      JAX/Flax implementation, see jeo_plugin/README.md
tests/           unit tests for the TensorFlow track
scripts/         plotting helper
docs/            these pages and the figures
results/         benchmark numbers and run configuration
```

## Data flow

```
TFRecord shard
  -> parse_example        s2 [T,H,W,10], elevation [H,W,3], climate [T,14], labels [H,W]
  -> normalize            per-modality scaling, climate z-scored over time
  -> augment (train only) synchronised flips, optical-only brightness jitter
  -> encode_labels        labels -> one-hot [H,W,9]
  -> batch, prefetch
  -> split_inputs_labels  ({s2, elevation, climate}, labels)
  -> MambaForestSegmenter -> [B,H,W,9] probabilities
  -> ComboLoss            scalar
```

Inside the network the batch passes through the temporal Mamba block, the two
spatial encoders, the fusion layer, three encoder stages, the spatial Mamba
bottleneck with SASF, three decoder stages with skip connections, and the
softmax head. [High-level Design](design.md) covers each of those in detail,
and [Programming Reference](reference.md) lists the exact signatures.

## Where to start

| You want to | Start in |
|---|---|
| Change the architecture | `model.py`, then `layers.py` for a new block |
| Change what the model is trained on | `data.py` |
| Change the objective | `losses.py` |
| Change the loop, metrics or schedule | `train.py` |
| Apply the model to new data | `predict.py`, [Mapping Your Own Region](apply.md) |
| Add a modality or a dataset | [Common Tasks](tasks.md) |
| Understand why something is built this way | [High-level Design](design.md) |

## Conventions

The code follows the [Coding Style](coding.md) page: two-space indentation,
Google-style docstrings that name tensor shapes, argument validation where a
wrong value would otherwise fail deep inside a framework call, and no
credentials or machine-specific paths in the source.

Before opening a change, run:

```bash
make test
make lint
```

New behaviour comes with a test in the matching file under `tests/`; the
[Testing](testing.md) page explains what the suite already covers and what
kind of test fits here.
