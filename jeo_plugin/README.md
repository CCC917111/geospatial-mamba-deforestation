Mamba-MTST: a Mamba encoder for JEO
=============
Welcome to the JAX track of the CNN-Mamba forest typology project!

[JEO](https://github.com/google-deepmind/jeo) is DeepMind's JAX framework for
geospatial model training: datasets, preprocessing ops, training loop and
evaluators are all selected from a config file, and models are looked up by
name under `jeo/models/`. This directory holds a second, independent
implementation of the same idea as the TensorFlow package — Mamba encoders in
place of transformer encoders for multi-temporal, multi-modal remote sensing —
written as a plug-in for that framework, plus tests that run without a JEO
checkout.

```
mamba_mtst.py                  -> jeo/models/mamba_mtst.py
configs/mamba_forty_v1.py      -> jeo/configs/forest/mamba_forty_v1.py
pp/forty_v1_ops.py             -> jeo/pp/forty_v1_ops.py
tests/test_mamba_mtst_shapes.py
```

## Model

`mamba_mtst.Model` replaces the transformer encoders of a multi-temporal,
multi-modal setup with Mamba encoders:

1. **Tokenisation.** Every modality is patchified with its own patch size —
   `(time, height, width)` — and projected to `d_model`. Modalities without a
   spatial extent (a climate series) or without a time axis (a terrain map) are
   handled by the same code path.
2. **Temporal encoder.** For each modality, one sequence per spatial patch is
   run through a stack of Mamba blocks and averaged over time.
3. **Fusion.** Per-modality summaries get a learned modality embedding and are
   summed; single-token modalities are broadcast over the spatial patches.
4. **Spatial encoder.** A second Mamba stack mixes the spatial patches.
5. **Head.** Mean-pooled classification, or a transposed-convolution decoder
   that upsamples the patch grid back to pixel resolution.

## Usage

With a JEO checkout:

```bash
git clone https://github.com/google-deepmind/jeo && cd jeo
cp /path/to/this/repo/jeo_plugin/mamba_mtst.py jeo/models/
cp /path/to/this/repo/jeo_plugin/pp/forty_v1_ops.py jeo/pp/
cp /path/to/this/repo/jeo_plugin/configs/mamba_forty_v1.py jeo/configs/forest/

python -m jeo.train \
    --config jeo/configs/forest/mamba_forty_v1.py:runlocal \
    --workdir /tmp/jeo/mamba_forty_v1
```

`:runlocal` runs five steps with a tiny model as a smoke test, `:quick` runs a
short reduced-size run, and no argument runs the full configuration.

The model file only imports `jax`, `flax` and `einops`, so it can also be used
on its own:

```bash
pip install jax flax einops
python jeo_plugin/tests/test_mamba_mtst_shapes.py
```

## Design notes

- **The depthwise convolution is causal.** It pads on the left only and
  convolves with `VALID`, so step `t` never sees `t + 1`; `SAME` padding would
  leak future time steps into the present one. A test perturbs the tail of a
  sequence and asserts that the head does not move.
- **Fusion does not depend on the order of `mods`.** A global modality such as
  the climate series can be listed anywhere, including first, and still reaches
  the fused representation; a test scales that modality and asserts the output
  changes.
- **The modality embedding is added per modality, before the sum,** so the
  network can tell the sources apart instead of receiving one embedding added
  several times to an already fused tensor.
- **Shape errors are explicit.** Patch divisibility, missing modalities and
  mismatched spatial grids raise `ValueError` with the offending modality named,
  rather than failing later inside `einops`.
- **The file is self-contained.** The positional-embedding helper is local, so
  the model depends only on jax, flax and einops and can be developed and
  tested without a JEO checkout.

## Tests

```bash
python jeo_plugin/tests/test_mamba_mtst_shapes.py
```

The suite covers the block and encoder output shapes, causality of the Mamba
block, both head types, gradient finiteness, and the fact that a global
modality listed first still influences the output.
