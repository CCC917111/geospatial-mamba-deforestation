"""Mamba U-Net for multimodal forest-type segmentation on ForTy v1."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers

from mamba_forest.layers import Mamba2Block, MultiModalFusion

# Channel counts of the ForTy v1 modalities used here.
OPTICAL_CHANNELS = 10   # Sentinel-2 bands
ELEVATION_CHANNELS = 3  # elevation, slope, aspect
CLIMATE_CHANNELS = 14   # climate variables per time step


class MambaForestSegmenter(tf.keras.Model):
  """Segmentation model with Mamba blocks on both the time and space axes.

  Pipeline:

  1. Optical time series [B, T, H, W, C] is folded to [B*H*W, T, C] and run
     through a temporal Mamba block; the last step is kept as the per-pixel
     temporal summary. The climate series [B, T, C] is handled the same way and
     broadcast over space.
  2. The three modality maps are projected to ``d_model`` channels and fused
     with :class:`MultiModalFusion`.
  3. The fused map is resized to ``bottleneck_size`` and encoded by a U-Net.
  4. At the bottleneck the spatial grid is flattened to a sequence and a
     spatial Mamba block mixes information across the whole tile.
  5. A U-Net decoder with skip connections produces per-pixel class scores,
     which are resized back to the input resolution.

  Args:
    num_classes: number of segmentation classes.
    d_model: channel width after the modality projections and fusion.
    dropout_rate: dropout used throughout the network.
    input_size: spatial size of the input tiles (assumed square).
    bottleneck_size: size the fused map is resized to before the U-Net; the
      encoder halves it three times, so 32 gives a 4x4 bottleneck grid.
  """

  def __init__(self, num_classes: int, d_model: int = 32,
               dropout_rate: float = 0.2, input_size: int = 128,
               bottleneck_size: int = 32, **kwargs):
    super().__init__(**kwargs)
    self.num_classes = num_classes
    self.d_model = d_model
    self.input_size = input_size
    self.bottleneck_size = bottleneck_size

    # --- Temporal encoders -------------------------------------------------
    self.temporal_mamba = Mamba2Block(
        d_model=OPTICAL_CHANNELS, d_state=8, expand=2, dropout_rate=dropout_rate)
    self.temporal_norm = layers.LayerNormalization(axis=-1)
    self.temporal_mamba_climate = Mamba2Block(
        d_model=CLIMATE_CHANNELS, d_state=4, expand=2, dropout_rate=dropout_rate)
    self.temporal_norm_climate = layers.LayerNormalization(axis=-1)

    # --- Per-modality projections -----------------------------------------
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
        layers.Dense(64, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(d_model, activation="relu"),
        layers.Dropout(dropout_rate),
    ], name="climate_proj")

    self.fusion_layer = MultiModalFusion(d_model, num_modalities=3, hidden_dim=64)
    self.spatial_adjust = layers.Conv2D(d_model, 1, padding="same")

    # --- U-Net encoder -----------------------------------------------------
    self.enc1 = self._encoder_block(64, dropout_rate)
    self.enc2 = self._encoder_block(128, dropout_rate)
    self.enc3 = self._encoder_block(256, dropout_rate)

    # --- Mamba bottleneck --------------------------------------------------
    self.bottleneck_conv = layers.Conv2D(512, 3, activation="relu", padding="same")
    self.spatial_mamba = Mamba2Block(
        d_model=512, d_state=16, expand=2, dropout_rate=dropout_rate)
    self.spatial_norm = layers.LayerNormalization(axis=-1)
    self.bottleneck_dropout = layers.Dropout(dropout_rate)

    # --- U-Net decoder -----------------------------------------------------
    self.up1 = self._decoder_block(256, dropout_rate)
    self.up2 = self._decoder_block(128, dropout_rate)
    self.up3 = self._decoder_block(64, dropout_rate)

    self.output_conv = tf.keras.Sequential([
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.Dropout(dropout_rate),
        layers.Conv2D(num_classes, 1, activation="softmax"),
    ], name="output_conv")

  @staticmethod
  def _encoder_block(out_channels: int, dropout_rate: float):
    return tf.keras.Sequential([
        layers.Conv2D(out_channels, 3, activation="relu", padding="same"),
        layers.Dropout(dropout_rate),
        layers.Conv2D(out_channels, 3, activation="relu", padding="same"),
        layers.Dropout(dropout_rate),
        layers.MaxPooling2D(2),
    ])

  @staticmethod
  def _decoder_block(out_channels: int, dropout_rate: float):
    return tf.keras.Sequential([
        layers.Conv2DTranspose(out_channels, 2, strides=2, padding="same"),
        layers.Conv2D(out_channels, 3, activation="relu", padding="same"),
        layers.Dropout(dropout_rate),
        layers.Conv2D(out_channels, 3, activation="relu", padding="same"),
        layers.Dropout(dropout_rate),
    ])

  # --- Modality encoders ---------------------------------------------------

  def _encode_optical(self, x, training: bool):
    """[B, T, H, W, C] -> [B, H, W, d_model]."""
    shape = tf.shape(x)
    batch, time, height, width = shape[0], shape[1], shape[2], shape[3]
    channels = x.shape[-1]  # static, needed by the dense layers

    # One sequence per pixel. The time axis has to be moved next to the
    # channels first: reshaping [B, T, H, W, C] directly would put
    # width-adjacent pixels of a single time step into one "sequence" instead
    # of the time steps of one pixel.
    x = tf.transpose(x, [0, 2, 3, 1, 4])  # [B, H, W, T, C]
    sequences = tf.reshape(x, (-1, time, channels))
    # The static channel dimension has to be restored after the reshape,
    # otherwise the layers inside the Mamba block cannot build their weights.
    sequences.set_shape([None, None, channels])
    sequences = self.temporal_mamba(sequences, training=training)
    summary = self.temporal_norm(sequences[:, -1, :])
    spatial = tf.reshape(summary, (batch, height, width, channels))
    spatial.set_shape([None, None, None, channels])
    return self.optical_proj(spatial, training=training)

  def _encode_elevation(self, x, training: bool):
    """[B, H, W, 3] -> [B, H, W, d_model]."""
    return self.elevation_proj(x, training=training)

  def _encode_climate(self, x, training: bool, height, width):
    """[B, T, C] -> [B, H, W, d_model] (broadcast over space)."""
    batch = tf.shape(x)[0]
    sequences = self.temporal_mamba_climate(x, training=training)
    summary = self.temporal_norm_climate(sequences[:, -1, :])
    features = self.climate_proj(summary, training=training)
    features = tf.reshape(features, (batch, 1, 1, self.d_model))
    return tf.tile(features, [1, height, width, 1])

  # --- Forward pass --------------------------------------------------------

  def call(self, inputs, training: bool = False):
    """Args: dict with keys ``s2``, ``elevation``, ``climate``.

    Returns per-pixel class probabilities [B, input_size, input_size,
    num_classes].
    """
    optical = inputs["s2"]
    elevation = inputs["elevation"]
    climate = inputs["climate"]

    spatial_shape = tf.shape(elevation)
    height, width = spatial_shape[1], spatial_shape[2]

    optical_feat = self._encode_optical(optical, training)
    elevation_feat = self._encode_elevation(elevation, training)
    climate_feat = self._encode_climate(climate, training, height, width)

    fused = self.fusion_layer(
        [optical_feat, elevation_feat, climate_feat], training=training)

    # The U-Net runs at a lower resolution to keep the model small; see the
    # "Known limitations" section of the README.
    x = tf.image.resize(fused, [self.bottleneck_size, self.bottleneck_size])
    x = self.spatial_adjust(x)

    e1 = self.enc1(x, training=training)
    e2 = self.enc2(e1, training=training)
    e3 = self.enc3(e2, training=training)

    # Bottleneck: flatten the grid into a sequence so the Mamba block can mix
    # information across the whole tile.
    b = self.bottleneck_conv(e3)
    b_shape = tf.shape(b)
    bh, bw = b_shape[1], b_shape[2]
    bc = b.shape[-1]  # static, required by the Mamba block
    b_seq = tf.reshape(b, [-1, bh * bw, bc])
    b_seq.set_shape([None, None, bc])
    b_seq = self.spatial_norm(self.spatial_mamba(b_seq, training=training))
    b = b + tf.reshape(b_seq, [-1, bh, bw, bc])
    b = self.bottleneck_dropout(b, training=training)

    d1 = tf.concat([self.up1(b, training=training), e2], axis=-1)
    d2 = tf.concat([self.up2(d1, training=training), e1], axis=-1)
    d3 = tf.concat([self.up3(d2, training=training), x], axis=-1)

    logits = self.output_conv(d3, training=training)
    return tf.image.resize(logits, [self.input_size, self.input_size])

  def get_config(self):
    config = super().get_config()
    config.update({
        "num_classes": self.num_classes,
        "d_model": self.d_model,
        "input_size": self.input_size,
        "bottleneck_size": self.bottleneck_size,
    })
    return config
