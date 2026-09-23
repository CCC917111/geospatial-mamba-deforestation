High-level Design
=============
Welcome to the CNN-Mamba forest typology high-level design page!

This page explains why the network is built the way it is, and gives the
equations behind the two Mamba stages.

![Architecture](figures/architecture.svg)

## The problem the architecture is solving

Three properties of the task drive every design decision.

1. **The signal is temporal, but the sequence is short.** Telling planted
   forest from natural forest is largely a question of how the canopy changes
   between seasons, yet each sample carries only four composites. A transformer
   spends quadratic attention on a four-step sequence and still has to be told
   how to weigh the modalities.
2. **The modalities are heterogeneous.** Optical imagery is a spatial time
   series, terrain is a static map, climate is a handful of numbers per season.
   Concatenating them at the input mixes signals of very different reliability:
   a cloudy tile makes the optical bands useless, while terrain stays valid.
3. **The classes of interest are rare.** Planted forest and tree crops together
   cover under 20% of the labelled pixels, and an objective dominated by pixel
   counts simply predicts the frequent classes.

The network answers these with, respectively, a selective state-space model, a
content-dependent fusion mechanism, and a region-level term in the loss.

## Selective state-space blocks

A state-space model keeps a hidden state and updates it linearly along the
sequence. What makes Mamba *selective* is that the update is produced from the
input, so the model can decide, per step and per state dimension, how much to
remember. `Mamba2Block` implements

```
alpha_t = exp(-exp(A) * softplus(W_dt s_delta(x_t)))        # forget gate
h_t     = alpha_t * h_{t-1} + (1 - alpha_t) * (s_B(x_t) * P x_t)
y_t     = (s_C(x_t) * h_t) P^T + D * conv(x)_t
```

where `s_delta`, `s_B` and `s_C` are small learned projections of the sequence,
`A` is a learned decay per state dimension, `P` is the shared projection
between the state space and the channels, and `conv` is a separable convolution
that mixes neighbouring steps before the scan. Because `alpha_t` lies in
`(0, 1)` and the update is convex, the recurrence cannot blow up, which matters
at the bottleneck where the sequence is 256 steps long.

The recurrence is evaluated with `tf.scan` rather than the fused CUDA kernel of
the reference implementation: O(L) sequential steps, portable to any device,
and irrelevant to the runtime at these sequence lengths.

The same block is used twice, on two different axes:

- **Temporally**, over the four seasons of each pixel. The optical tile is
  transposed to `[B, H, W, T, C]` before being folded to `[B*H*W, T, C]`, so a
  sequence really is the time series of one pixel rather than a row of
  neighbouring pixels. The last state is the temporal summary.
- **Spatially**, over the flattened 16 x 16 bottleneck grid, so each patch can
  reach every other patch in linear time.

## Structure-Aware State Fusion

Flattening a 2D grid into a 1D sequence keeps horizontal adjacency and destroys
vertical adjacency: pixel `(i, j)` and pixel `(i+1, j)` are 16 steps apart in
the scan. SASF restores it after the scan,

```
h_fused(i) = h(i) + sum_{n in N(i)} w_n * h(n),
```

over the four direct neighbours, with `w_n` learned per direction and channel
and initialised small so the layer starts close to the identity. Borders are
zero-padded, so a pixel on the edge simply sees fewer neighbours. Each state
stays its own leading term, and the Mamba block passes the convolutional
features through its internal residual, so they reach the decoder — in
normalised form — even if the scan contributes little early in training.

## Dynamic multi-modal fusion

The fusion layer pools each modality to a global descriptor, concatenates the
descriptors, and passes them through a two-layer MLP with a softmax over the
modality axis. This produces weights that depend on the content of the sample
rather than a single blend learned once for the whole dataset: the network can
down-weight the optical branch on a cloudy tile and lean on terrain instead.
The weighted sum then passes through two 1 x 1 convolutions and layer
normalisation, which lets channels of different modalities interact after the
mixing.

## Resolution and the U-Net

The fused map enters the U-Net at the full 128 x 128 resolution, and the three
encoder stages take it to 16 x 16 while the width grows 32 -> 64 -> 128 -> 256.
Running the encoder at full resolution is what keeps class boundaries sharp:
every downsampling step is paired with a skip connection that returns the
detail to the decoder, and the deepest skip is the fused map itself, so the
final decoder stage still sees un-pooled features.

The bottleneck at 16 x 16 is where the spatial Mamba block sits. That choice is
deliberate: a 256-step sequence is long enough for global context to be
meaningful and short enough for the scan to be cheap, whereas scanning the full
128 x 128 grid would mean a 16384-step sequence for no additional receptive
field, since the bottleneck is already global.

## The objective

```
L = alpha * L_WCCE + (1 - alpha) * L_Dice,   alpha = 0.4
```

The cross-entropy term is weighted per class and smoothed by 0.05. Class
weighting addresses the imbalance directly at the pixel level; label smoothing
keeps the network from becoming over-confident on the dominant classes, which
in practice is what makes the rare-class gradients survive.

The Dice term measures region overlap per class and is therefore insensitive to
how many pixels a class occupies — a small planted-forest patch counts as much
as a large natural-forest one. It is averaged only over the classes present in
the batch: a class that appears in neither the labels nor the prediction would
otherwise score a perfect `smooth / smooth = 1` and reward the network for
ignoring it.

The 0.4 / 0.6 split puts the majority of the weight on the region term, which
is what the benchmark ultimately measures, while keeping enough pixel-level
supervision for the output to stay calibrated.

## Training loop

The loop is written explicitly rather than through `model.fit`, because the
metrics of interest are computed from a confusion matrix accumulated across
batches. Per epoch: a fresh random subset of training shards, one pass, then
evaluation on the fixed validation shards. The learning rate follows a cosine
decay to 10% of its initial value, gradients are clipped to a global norm of
1.0, and mixed precision is handled by a loss-scaling optimizer. The checkpoint
with the best Overall F1 is kept, and training stops early after `--patience`
epochs without improvement.
