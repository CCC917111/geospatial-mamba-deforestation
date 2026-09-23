# Mamba-MTST: a Mamba encoder for JEO

[JEO](https://github.com/google-deepmind/jeo) is DeepMind's JAX framework for
geospatial model training: datasets, preprocessing ops, training loop and
evaluators are all selected from a config file, and models are looked up by
name under `jeo/models/`. This directory contains the three files needed to
train a Mamba-based model inside that framework, plus tests that run without a
JEO checkout.

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

The tests cover output shapes for both heads, gradient finiteness, the fact
that a global modality listed first still reaches the fused representation, and
that the block is causal — a time step must not be influenced by later ones,
which an earlier `SAME`-padded convolution silently violated.

## Differences from the version developed during the course project

- The depthwise convolution is causal (left padding) instead of `SAME`.
- Fusion no longer depends on the order of `mods`: a global modality listed
  first used to be dropped with a warning.
- The modality embedding is added to each modality before the sum instead of
  being added several times to the already fused tensor.
- Patch divisibility, missing modalities and mismatched spatial sizes raise
  explicit errors instead of failing later inside `einops`.
- The positional embedding helper is local, so the file no longer depends on
  `jeo.components`; the unused `jeo.tools.checkpointing` import is gone.
