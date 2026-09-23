"""Efficient Multi-Modal CNN-Mamba segmentation network for ForTy v1.

The network combines convolutional encoders for local spatial structure with
selective state-space (Mamba) blocks for linear-time temporal and global
context modelling:

  1. Parallel modality encoding. The optical time series [B, T, H, W, C] is
     folded into one sequence per pixel and passed through a temporal Mamba
     block, which compresses the seasonal dynamics into a single feature map.
     Elevation goes through a small 2D convolutional stack, and the climate
     series through an MLP whose embedding is broadcast over space.
  2. Dynamic multi-modal fusion. An MLP-based attention network computes one
     softmax weight per modality and per sample, and the weighted sum is
     refined by 1x1 convolutions.
  3. U-Net encoder. Three encoder stages take the fused 128x128 map down to
     16x16 while the channel width grows 32 -> 64 -> 128 -> 256.
  4. Spatial Mamba bottleneck with SASF. The 16x16x512 bottleneck is flattened
     into a sequence of length 256 and scanned by a spatial Mamba block, then
     Structure-Aware State Fusion re-introduces the 2D adjacency that the
     flattening destroys.
  5. Multi-stage decoder. Three decoder stages return to 128x128, each
     concatenating the matching encoder feature map, followed by a
     convolutional classification head with softmax.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers

from mamba_forest.layers import (DynamicMultiModalFusion, Mamba2Block,
                                 StructureAwareStateFusion)

# Channel counts of the ForTy v1 modalities.
OPTICAL_CHANNELS = 10   # Sentinel-2 bands
ELEVATION_CHANNELS = 3  # elevation, slope, aspect
CLIMATE_CHANNELS = 14   # climate variables per time step


class MambaForestSegmenter(tf.keras.Model):
  """CNN-Mamba segmenter for multi-modal forest typology mapping.

  Args:
    num_classes: number of segmentation classes.
    d_model: channel width after the modality projections and fusion.
    dropout_rate: dropout used throughout the network.
    d_state_temporal: state dimension of the temporal Mamba block.
    d_state_spatial: state dimension of the spatial Mamba block.
    bottleneck_channels: channel width of the bottleneck feature map.
  """

  def __init__(self, num_classes: int, d_model: int = 32,
               dropout_rate: float = 0.2, d_state_temporal: int = 8,
               d_state_spatial: int = 16, bottleneck_channels: int = 512,
               **kwargs):
    super().__init__(**kwargs)
    self.num_classes = num_classes
    self.d_model = d_model
    self.dropout_rate = dropout_rate
    self.d_state_temporal = d_state_temporal
    self.d_state_spatial = d_state_spatial
    self.bottleneck_channels = bottleneck_channels

    # --- Temporal encoder ---------------------------------------------------
    self.temporal_mamba = Mamba2Block(
        d_model=OPTICAL_CHANNELS, d_state=d_state_temporal, expand=2,
        dropout_rate=dropout_rate)
    self.temporal_norm = layers.LayerNormalization(axis=-1)

    # --- Per-modality projections -------------------------------------------
    self.optical_proj = tf.keras.Sequential([
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.Conv2D(d_model, 1, activation="relu", padding="same"),
        layers.Dropout(dropout_rate),
    ], name="optical_proj")
    self.elevation_proj = tf.keras.Sequential([
        layers.Conv2D(16, 3, activation="relu", padding="same"),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.Conv2D(d_model, 1, activation="relu", padding="same"),
        layers.Dropout(dropout_rate),
    ], name="elevation_proj")
    self.climate_proj = tf.keras.Sequential([
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(d_model, activation="relu"),
        layers.Dropout(dropout_rate),
    ], name="climate_proj")

    # --- Dynamic multi-modal fusion -----------------------------------------
    self.fusion_layer = DynamicMultiModalFusion(
        d_model, num_modalities=3, hidden_dim=64)

    # --- U-Net encoder: 128 -> 64 -> 32 -> 16 -------------------------------
    self.enc1 = self._encoder_block(64, dropout_rate, "enc1")
    self.enc2 = self._encoder_block(128, dropout_rate, "enc2")
    self.enc3 = self._encoder_block(256, dropout_rate, "enc3")

    # --- Spatial Mamba bottleneck with SASF ---------------------------------
    self.bottleneck_conv = layers.Conv2D(
        bottleneck_channels, 3, activation="relu", padding="same")
    self.spatial_mamba = Mamba2Block(
        d_model=bottleneck_channels, d_state=d_state_spatial, expand=2,
        dropout_rate=dropout_rate)
    self.spatial_norm = layers.LayerNormalization(axis=-1)
    self.sasf = StructureAwareStateFusion(name="sasf")
    self.bottleneck_dropout = layers.Dropout(dropout_rate)

    # --- Multi-stage decoder: 16 -> 32 -> 64 -> 128 -------------------------
    self.up1 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")
    self.dec1 = self._decoder_block(256, dropout_rate, "dec1")
    self.up2 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")
    self.dec2 = self._decoder_block(128, dropout_rate, "dec2")
    self.up3 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")
    self.dec3 = self._decoder_block(64, dropout_rate, "dec3")

    # --- Classification head ------------------------------------------------
    self.head_conv = layers.Conv2D(32, 3, activation="relu", padding="same")
    self.head_dropout = layers.Dropout(dropout_rate)
    # Kept in float32 so that the softmax and the loss stay numerically stable
    # when the rest of the network runs in mixed precision.
    self.head_output = layers.Conv2D(
        num_classes, 1, activation="softmax", dtype="float32")

  @staticmethod
  def _encoder_block(out_channels: int, dropout_rate: float, name: str):
    """Two 3x3 convolutions with dropout, then 2x max pooling."""
    return tf.keras.Sequential([
        layers.Conv2D(out_channels, 3, activation="relu", padding="same"),
        layers.Dropout(dropout_rate),
        layers.Conv2D(out_channels, 3, activation="relu", padding="same"),
        layers.Dropout(dropout_rate),
        layers.MaxPooling2D(2),
    ], name=name)

  @staticmethod
  def _decoder_block(out_channels: int, dropout_rate: float, name: str):
    """Two 3x3 convolutions with dropout, applied after the skip concat."""
    return tf.keras.Sequential([
        layers.Conv2D(out_channels, 3, activation="relu", padding="same"),
        layers.Dropout(dropout_rate),
        layers.Conv2D(out_channels, 3, activation="relu", padding="same"),
        layers.Dropout(dropout_rate),
    ], name=name)

  # --- Modality encoders ----------------------------------------------------

  def _encode_optical(self, x, training: bool):
    """[B, T, H, W, C] -> [B, H, W, d_model]."""
    shape = tf.shape(x)
    batch, time, height, width = shape[0], shape[1], shape[2], shape[3]
    channels = x.shape[-1]  # static, needed by the dense layers

    # One sequence per pixel. The time axis is moved next to the channels
    # first: reshaping [B, T, H, W, C] directly would put width-adjacent pixels
    # of a single time step into one sequence instead of the time steps of one
    # pixel.
    x = tf.transpose(x, [0, 2, 3, 1, 4])  # [B, H, W, T, C]
    sequences = tf.reshape(x, (-1, time, channels))
    # The static channel dimension has to be restored after the reshape,
    # otherwise the layers inside the Mamba block cannot build their weights.
    sequences.set_shape([None, None, channels])
    sequences = self.temporal_mamba(sequences, training=training)
    # The last state summarises the whole season sequence.
    summary = self.temporal_norm(sequences[:, -1, :])
    spatial = tf.reshape(summary, (batch, height, width, channels))
    spatial.set_shape([None, None, None, channels])
    return self.optical_proj(spatial, training=training)

  def _encode_elevation(self, x, training: bool):
    """[B, H, W, 3] -> [B, H, W, d_model]."""
    return self.elevation_proj(x, training=training)

  def _encode_climate(self, x, training: bool, height, width):
    """[B, T, C] -> [B, H, W, d_model], broadcast over space."""
    batch = tf.shape(x)[0]
    features = self.climate_proj(x, training=training)
    features = tf.reshape(features, (batch, 1, 1, self.d_model))
    return tf.tile(features, [1, height, width, 1])

  # --- Forward pass ---------------------------------------------------------

  def call(self, inputs, training: bool = False):
    """Args: dict with keys ``s2``, ``elevation``, ``climate``.

    Returns per-pixel class probabilities [B, H, W, num_classes] at the
    resolution of the input tile.
    """
    optical = inputs["s2"]
    elevation = inputs["elevation"]
    climate = inputs["climate"]

    spatial_shape = tf.shape(elevation)
    height, width = spatial_shape[1], spatial_shape[2]

    optical_feat = self._encode_optical(optical, training)
    elevation_feat = self._encode_elevation(elevation, training)
    climate_feat = self._encode_climate(climate, training, height, width)

    # [B, H, W, d_model]
    x = self.fusion_layer(
        [optical_feat, elevation_feat, climate_feat], training=training)

    e1 = self.enc1(x, training=training)    # [B, H/2,  W/2,   64]
    e2 = self.enc2(e1, training=training)   # [B, H/4,  W/4,  128]
    e3 = self.enc3(e2, training=training)   # [B, H/8,  W/8,  256]

    # Bottleneck: flatten the grid into a sequence so the Mamba block can mix
    # information across the whole tile in linear time, then restore the 2D
    # neighbourhood with SASF.
    b = self.bottleneck_conv(e3)            # [B, H/8, W/8, 512]
    b_shape = tf.shape(b)
    bh, bw = b_shape[1], b_shape[2]
    bc = b.shape[-1]  # static, required by the Mamba block
    b_seq = tf.reshape(b, [-1, bh * bw, bc])
    b_seq.set_shape([None, None, bc])
    b_seq = self.spatial_norm(self.spatial_mamba(b_seq, training=training))
    b_map = tf.reshape(b_seq, [-1, bh, bw, bc])
    b_map.set_shape([None, None, None, bc])
    # SASF already keeps the state of each patch as its own leading term, and
    # the Mamba block carries the convolutional features through its residual,
    # so no further skip is needed here.
    b = self.bottleneck_dropout(self.sasf(b_map), training=training)

    d1 = self.dec1(tf.concat([self.up1(b), e2], axis=-1), training=training)
    d2 = self.dec2(tf.concat([self.up2(d1), e1], axis=-1), training=training)
    d3 = self.dec3(tf.concat([self.up3(d2), x], axis=-1), training=training)

    head = self.head_dropout(self.head_conv(d3), training=training)
    return self.head_output(head)

  def get_config(self):
    config = super().get_config()
    config.update({
        "num_classes": self.num_classes,
        "d_model": self.d_model,
        "dropout_rate": self.dropout_rate,
        "d_state_temporal": self.d_state_temporal,
        "d_state_spatial": self.d_state_spatial,
        "bottleneck_channels": self.bottleneck_channels,
    })
    return config
