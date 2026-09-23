"""Mamba-MTST: a Mamba multi-modal temporal/spatial encoder in JAX + Flax.

The model is a drop-in replacement for the transformer encoders used in the
JEO remote-sensing framework (https://github.com/google-deepmind/jeo): every
modality is patchified into tokens, a Mamba encoder runs along the time axis of
each spatial patch, the per-modality summaries are fused, and a second Mamba
encoder runs along the spatial axis before the task head.

The file is self-contained (it only needs jax, flax and einops) so that the
model can be unit-tested without a JEO checkout; see
``tests/test_mamba_mtst_shapes.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp


def learned_positional_embedding(module: nn.Module, name: str, length: int,
                                 width: int, dtype=jnp.float32):
  """Returns a learned positional embedding of shape [1, length, width]."""
  init = nn.initializers.normal(stddev=1 / jnp.sqrt(width))
  return module.param(name, init, (1, length, width), dtype)


class MambaBlock(nn.Module):
  """Selective state-space block (Mamba).

  Sequence of operations: input projection into a processed branch and a gate,
  a causal depthwise convolution, the selective scan, gating and an output
  projection.

  Attributes:
    d_model: input and output channel dimension.
    d_state: state dimension N of the SSM.
    d_conv: kernel width of the causal convolution.
    expand: expansion factor of the inner dimension.
    dt_rank: rank of the low-rank delta projection, or 'auto'.
  """

  d_model: int
  d_state: int = 16
  d_conv: int = 4
  expand: int = 2
  dt_rank: Any = "auto"

  @nn.compact
  def __call__(self, x, train: bool = False):
    del train  # The block has no train-specific behaviour.
    d_inner = self.expand * self.d_model
    dt_rank = (max(1, self.d_model // 16) if self.dt_rank == "auto"
               else self.dt_rank)

    x_and_res = nn.Dense(d_inner * 2, use_bias=False, name="in_proj")(x)
    x_branch, res_branch = jnp.split(x_and_res, 2, axis=-1)

    # Causal convolution: pad on the left only, so that step t never sees
    # steps t + 1, ... ('SAME' padding would leak future time steps).
    x_padded = jnp.pad(x_branch, ((0, 0), (self.d_conv - 1, 0), (0, 0)))
    x_conv = nn.Conv(
        features=d_inner,
        kernel_size=(self.d_conv,),
        feature_group_count=d_inner,
        padding="VALID",
        use_bias=True,
        name="conv1d")(x_padded)
    x_conv = nn.silu(x_conv)

    y = self._selective_scan(x_conv, d_inner, dt_rank)
    y = y * nn.silu(res_branch)
    return nn.Dense(self.d_model, use_bias=False, name="out_proj")(y)

  def _selective_scan(self, x, d_inner: int, dt_rank: int):
    """Input-dependent discretisation followed by the linear recurrence."""
    x_dbl = nn.Dense(dt_rank + self.d_state * 2, use_bias=False,
                     name="x_proj")(x)
    delta, b_mat, c_mat = jnp.split(
        x_dbl, [dt_rank, dt_rank + self.d_state], axis=-1)

    delta = nn.softplus(nn.Dense(d_inner, use_bias=True, name="dt_proj")(delta))
    a_mat = -jnp.exp(
        self.param("A", nn.initializers.lecun_normal(), (d_inner, self.d_state)))
    d_vec = self.param("D", nn.initializers.ones, (d_inner,))

    batch_size = x.shape[0]
    delta_a = jnp.einsum("bld,dn->bldn", delta, a_mat)
    delta_b_x = jnp.einsum("bld,bln,bld->bldn", delta, b_mat, x)

    def scan_fn(h, inputs):
      delta_a_t, delta_b_x_t, c_t = inputs
      h = jnp.exp(delta_a_t) * h + delta_b_x_t
      return h, jnp.einsum("bdn,bn->bd", h, c_t)

    _, y_seq = jax.lax.scan(
        scan_fn,
        jnp.zeros((batch_size, d_inner, self.d_state)),
        (jnp.transpose(delta_a, (1, 0, 2, 3)),
         jnp.transpose(delta_b_x, (1, 0, 2, 3)),
         jnp.transpose(c_mat, (1, 0, 2))))

    y = jnp.transpose(y_seq, (1, 0, 2))  # [batch, seq_len, d_inner]
    return y + x * d_vec


class MambaEncoder(nn.Module):
  """Stack of pre-norm Mamba blocks, each followed by an MLP block."""

  depth: int
  d_model: int
  d_state: int = 16
  d_conv: int = 4
  expand: int = 2
  mlp_ratio: int = 4
  dropout: float = 0.1

  @nn.compact
  def __call__(self, x, train: bool = False):
    for i in range(self.depth):
      y = nn.LayerNorm(name=f"ln_mamba_{i}")(x)
      y = MambaBlock(
          d_model=self.d_model, d_state=self.d_state, d_conv=self.d_conv,
          expand=self.expand, name=f"mamba_block_{i}")(y, train=train)
      x = x + nn.Dropout(rate=self.dropout)(y, deterministic=not train)

      y = nn.LayerNorm(name=f"ln_mlp_{i}")(x)
      y = nn.Dense(self.d_model * self.mlp_ratio, name=f"mlp_dense_in_{i}")(y)
      y = nn.gelu(y)
      y = nn.Dense(self.d_model, name=f"mlp_dense_out_{i}")(y)
      x = x + nn.Dropout(rate=self.dropout)(y, deterministic=not train)
    return nn.LayerNorm(name="ln_final")(x)


class Model(nn.Module):
  """Multi-modal Mamba model for forest classification and segmentation.

  Attributes:
    mods: names of the input modalities, used as keys of the input dict.
    patch_sizes: patch size per modality, either (height, width) or
      (time, height, width).
    num_classes: number of output classes.
    d_model: token width.
    d_state: SSM state dimension.
    temporal_depth: number of Mamba blocks along the time axis.
    spatial_depth: number of Mamba blocks along the spatial axis.
    dropout: dropout rate.
    head: 'classification' for one label per tile, 'segmentation' for a
      per-pixel map.
  """

  mods: Sequence[str]
  patch_sizes: dict[str, Sequence[int]]
  num_classes: int
  d_model: int = 256
  d_state: int = 16
  temporal_depth: int = 4
  spatial_depth: int = 2
  dropout: float = 0.1
  head: str = "classification"

  @nn.compact
  def __call__(self, inputs, *, train: bool = False):
    if not self.mods:
      raise ValueError("At least one modality is required.")
    missing = [m for m in self.mods if m not in inputs]
    if missing:
      raise ValueError(f"Missing modalities in the input dict: {missing}")

    out = {}
    tokens, patched_shapes = self._tokenize(inputs)
    out["tokens"] = tokens

    temporal_features = self._encode_temporal(tokens, train)
    out["temporal_features"] = temporal_features

    spatial_features = self._encode_spatial(
        temporal_features, patched_shapes, train)
    out["spatial_features"] = spatial_features

    logits = self._head(spatial_features, patched_shapes, train)
    out["logits"] = logits
    return logits, out

  def _tokenize(self, inputs):
    """Patchifies every modality into [batch, time, space, d_model] tokens."""
    tokens, patched_shapes = {}, {}
    for mod in self.mods:
      x, patch_size = self._as_5d(inputs[mod], self.patch_sizes[mod])
      _, t, h, w, _ = x.shape
      pt, ph, pw = patch_size
      for size, patch, axis in ((t, pt, "time"), (h, ph, "height"),
                                (w, pw, "width")):
        if size % patch:
          raise ValueError(
              f"Modality {mod!r}: {axis} {size} is not divisible by patch {patch}")
      patched_shapes[mod] = (t // pt, h // ph, w // pw)
      x = einops.rearrange(
          x, "b (t pt) (h ph) (w pw) c -> b t (h w) (pt ph pw c)",
          pt=pt, ph=ph, pw=pw)
      tokens[mod] = nn.Dense(self.d_model, name=f"proj_{mod}")(x)
    return tokens, patched_shapes

  def _encode_temporal(self, tokens, train: bool):
    """Runs a Mamba encoder over the time axis of every spatial patch."""
    temporal_features = {}
    for mod in self.mods:
      x = tokens[mod]
      batch, t, s, d = x.shape
      x = einops.rearrange(x, "b t s d -> (b s) t d")
      x = x + learned_positional_embedding(
          self, f"pe_temporal_{mod}", t, d, x.dtype)
      x = MambaEncoder(
          depth=self.temporal_depth, d_model=d, d_state=self.d_state,
          dropout=self.dropout, name=f"temporal_encoder_{mod}")(x, train=train)
      x = jnp.mean(x, axis=1)  # average over time
      temporal_features[mod] = einops.rearrange(x, "(b s) d -> b s d", b=batch)
    return temporal_features

  def _encode_spatial(self, temporal_features, patched_shapes, train: bool):
    """Fuses the modalities and mixes the spatial patches with Mamba.

    Modalities that carry a single token (e.g. a climate series without spatial
    extent) are broadcast over the spatial patches of the other modalities, so
    the fusion does not depend on the order of ``mods``.
    """
    spatial_sizes = {m: f.shape[1] for m, f in temporal_features.items()}
    grids = {m: tuple(patched_shapes[m][1:]) for m in self.mods
             if spatial_sizes[m] > 1}
    if len(set(grids.values())) > 1:
      raise ValueError(f"Modalities have different patch grids: {grids}")
    num_patches = next(iter(grids.values()), (1, 1))
    num_patches = num_patches[0] * num_patches[1]

    fused = None
    for mod in self.mods:
      features = temporal_features[mod]
      if features.shape[1] == 1 and num_patches > 1:
        features = jnp.tile(features, (1, num_patches, 1))
      # One learned embedding per modality, added before the sum so that the
      # model can tell the modalities apart.
      features = features + self.param(
          f"modality_embed_{mod}", nn.initializers.normal(stddev=0.02),
          (1, 1, self.d_model))
      fused = features if fused is None else fused + features

    fused = fused + learned_positional_embedding(
        self, "pe_spatial", num_patches, self.d_model, fused.dtype)
    return MambaEncoder(
        depth=self.spatial_depth, d_model=self.d_model, d_state=self.d_state,
        dropout=self.dropout, name="spatial_encoder")(fused, train=train)

  def _head(self, features, patched_shapes, train: bool):
    if self.head == "classification":
      x = jnp.mean(features, axis=1)
      x = nn.LayerNorm(name="head_ln")(x)
      x = nn.gelu(nn.Dense(self.d_model, name="head_dense")(x))
      x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
      return nn.Dense(self.num_classes, name="head_output")(x)

    if self.head == "segmentation":
      # Restore the patch grid of the first spatially resolved modality and
      # upsample it back to pixel resolution.
      mod = next((m for m in self.mods if patched_shapes[m][1] > 1
                  or patched_shapes[m][2] > 1), self.mods[0])
      patch_size = self.patch_sizes[mod]
      ph, pw = (patch_size[-2], patch_size[-1])
      _, h_p, w_p = patched_shapes[mod]
      if ph % 4 or pw % 4:
        raise ValueError(
            "The segmentation head upsamples by 2 x 2 x (patch / 4), so the "
            f"spatial patch size must be a multiple of 4; got {(ph, pw)}")

      x = features.reshape(features.shape[0], h_p, w_p, -1)
      x = nn.gelu(nn.Conv(self.d_model * 4, (1, 1), name="head_conv")(x))
      x = nn.gelu(nn.ConvTranspose(
          self.d_model * 2, (2, 2), strides=(2, 2), name="head_deconv_1")(x))
      x = nn.gelu(nn.ConvTranspose(
          self.d_model, (2, 2), strides=(2, 2), name="head_deconv_2")(x))
      x = nn.gelu(nn.ConvTranspose(
          self.d_model // 2, (ph // 4, pw // 4), strides=(ph // 4, pw // 4),
          name="head_deconv_3")(x))
      return nn.Conv(self.num_classes, (1, 1), name="head_output")(x)

    raise ValueError(f"Unknown head type: {self.head}")

  @staticmethod
  def _as_5d(x, patch_size):
    """Normalises an input to [batch, time, height, width, channels]."""
    patch_size = tuple(patch_size)
    if len(patch_size) == 2:
      patch_size = (1,) + patch_size
    if x.ndim == 3:      # [batch, time, channels] - no spatial extent
      x = x[:, :, None, None, :]
    elif x.ndim == 4:    # [batch, height, width, channels] - no time axis
      x = x[:, None, :, :, :]
    elif x.ndim != 5:
      raise ValueError(f"Expected a 3-, 4- or 5-D input, got shape {x.shape}")
    return x, patch_size
