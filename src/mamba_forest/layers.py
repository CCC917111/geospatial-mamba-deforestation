"""Building blocks of the CNN-Mamba segmentation network.

The module provides the three components the architecture is built from:

  Mamba2Block                 selective state-space block used along the time
                              axis (one sequence per pixel) and along the
                              flattened spatial axis at the bottleneck
  StructureAwareStateFusion   re-injects 2D adjacency after the 1D Mamba scan
  DynamicMultiModalFusion     content-dependent attention over the modalities
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


class Mamba2Block(layers.Layer):
  """Selective state-space block.

  The block keeps a state of size ``d_state`` per sequence and updates it with
  a decay that is produced from the sequence itself, which is what makes the
  state space *selective*: a slow signal (a stable canopy) can be held while a
  fast one (a clear-cut between two seasons) is reacted to.

  Per time step t, with state h of size R = ``d_state``:

      alpha_t = exp(-exp(A) * softplus(W_dt s_delta(x_t)))   (R,)
      h_t     = alpha_t * h_{t-1} + (1 - alpha_t) * (s_B(x_t) * P x_t)
      y_t     = (s_C(x_t) * h_t) P^T + D * conv(x)_t

  where ``P`` is the shared state <-> channel projection, ``s_delta``, ``s_B``
  and ``s_C`` are the small learned projections that generate the selective SSM
  parameters, and ``conv`` is a separable convolution that extracts short-range
  patterns before the scan. The recurrence is evaluated with ``tf.scan``, so it
  costs O(L) sequential steps and runs on any device.

  Args:
    d_model: channel dimension of the input and of the output.
    d_state: SSM state dimension R.
    expand: expansion factor E; the block works internally with E * d_model.
    dropout_rate: dropout applied to the block output.
  """

  def __init__(self, d_model: int, d_state: int = 16, expand: int = 2,
               dropout_rate: float = 0.1, **kwargs):
    super().__init__(**kwargs)
    self.d_model = int(d_model)
    self.d_state = int(d_state)
    self.expand = int(expand)
    self.d_inner = self.expand * self.d_model
    self.dropout_rate = dropout_rate

    # Input projection producing the processed branch and the gate.
    self.in_proj = layers.Dense(self.d_inner * 2, use_bias=True)

    # Short-range mixing before the scan.
    self.conv = layers.SeparableConv1D(
        filters=self.d_inner, kernel_size=3, padding="same", use_bias=True)

    # Input-dependent SSM parameters (all of size d_state).
    self.s_delta = layers.Dense(self.d_state, use_bias=True)
    self.s_B = layers.Dense(self.d_state, use_bias=True)
    self.s_C = layers.Dense(self.d_state, use_bias=True)
    self.dt_proj = layers.Dense(self.d_state, use_bias=True)

    self.out_proj = layers.Dense(self.d_model, use_bias=False)
    self.norm = layers.LayerNormalization(axis=-1, epsilon=1e-6)
    self.dropout = layers.Dropout(self.dropout_rate)

  def build(self, input_shape):
    # Decay parameter per state dimension, initialised negative so that
    # exp(A_raw) is small and the initial state decays slowly.
    self.A_raw = self.add_weight(
        name="A_raw", shape=(self.d_state,),
        initializer=tf.keras.initializers.RandomUniform(-3.0, -1.0),
        trainable=True)
    # Shared low-rank projection between the state space and the channels.
    self.state_to_channel = self.add_weight(
        name="state_to_channel", shape=(self.d_state, self.d_inner),
        initializer="glorot_uniform", trainable=True)
    # Per-channel skip scale on the convolution path.
    self.D = self.add_weight(
        name="D", shape=(self.d_inner,),
        initializer=tf.keras.initializers.Ones(), trainable=True)
    super().build(input_shape)

  def call(self, x, training: bool = False):
    """Applies the block to ``x`` of shape [B, L, d_model] (residual included)."""
    residual = x
    x_ln = self.norm(x)

    x_proc, gate = tf.split(self.in_proj(x_ln), 2, axis=-1)  # [B, L, d_inner]
    conv_act = tf.nn.silu(self.conv(x_proc))                 # [B, L, d_inner]

    # Input-dependent SSM parameters, all [B, L, R].
    s_delta = self.s_delta(conv_act)
    s_B = self.s_B(conv_act)
    s_C = self.s_C(conv_act)
    dt = tf.nn.softplus(self.dt_proj(s_delta))

    # Per-step decay: larger dt or larger A means faster forgetting.
    alpha = tf.exp(-tf.exp(self.A_raw) * dt)                 # [B, L, R]

    # Project the channel dimension down to the state dimension and modulate
    # it with the input-dependent B.
    conv_to_state = tf.tensordot(
        conv_act, tf.transpose(self.state_to_channel), axes=[[2], [0]])
    u = s_B * conv_to_state                                  # [B, L, R]

    # Sequential state update (time-major for tf.scan). The recurrence is kept
    # in float32 even under a mixed-precision policy: a few hundred sequential
    # multiply-accumulates in float16 would lose more precision than the step
    # saves in time.
    batch_size = tf.shape(x)[0]
    h_0 = tf.zeros((batch_size, self.d_state), dtype=tf.float32)

    def step(h_prev, inputs):
      alpha_t, u_t = inputs
      return alpha_t * h_prev + (1.0 - alpha_t) * u_t

    h_all = tf.scan(
        step,
        elems=(tf.cast(tf.transpose(alpha, [1, 0, 2]), tf.float32),
               tf.cast(tf.transpose(u, [1, 0, 2]), tf.float32)),
        initializer=h_0,
        parallel_iterations=10)
    h_all = tf.cast(tf.transpose(h_all, [1, 0, 2]), conv_act.dtype)  # [B, L, R]

    # Read the state out with the input-dependent C, lift it back to the
    # channel dimension and add the scaled convolution path.
    y = tf.tensordot(h_all * s_C, self.state_to_channel, axes=[[2], [0]])
    y = y + conv_act * tf.reshape(self.D, (1, 1, -1))        # [B, L, d_inner]

    out = self.out_proj(y * tf.nn.silu(gate))                # [B, L, d_model]
    out = self.dropout(out, training=training)
    return residual + out

  def get_config(self):
    config = super().get_config()
    config.update({
        "d_model": self.d_model,
        "d_state": self.d_state,
        "expand": self.expand,
        "dropout_rate": self.dropout_rate,
    })
    return config


class StructureAwareStateFusion(layers.Layer):
  """Structure-Aware State Fusion (SASF) over a 2D grid of states.

  Flattening a feature map into a sequence destroys vertical adjacency: two
  pixels that are neighbours in the image are a full row apart in the scan
  order, so the state of one barely reaches the other. SASF restores that 2D
  inductive bias by mixing each state with its four direct neighbours,

      h_fused(i) = h(i) + sum_{n in N(i)} w_n * h(n),

  where ``N(i)`` is the up/down/left/right neighbourhood and the four weights
  ``w_n`` are learned per direction and channel and initialised small, so the
  layer starts close to the identity. Borders are handled by zero padding, i.e.
  a missing neighbour contributes nothing.

  Input and output are feature maps of shape [B, H, W, C].
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def build(self, input_shape):
    channels = int(input_shape[-1])
    self.channels = channels
    # One weight per direction and channel: up, down, left, right.
    self.neighbour_weights = self.add_weight(
        name="neighbour_weights", shape=(4, channels),
        initializer=tf.keras.initializers.Constant(0.05), trainable=True)
    super().build(input_shape)

  def call(self, x):
    # Shift the map in the four directions; the vacated row/column is zero, so
    # pixels on the border simply see fewer neighbours.
    up = tf.pad(x[:, 1:, :, :], [[0, 0], [0, 1], [0, 0], [0, 0]])
    down = tf.pad(x[:, :-1, :, :], [[0, 0], [1, 0], [0, 0], [0, 0]])
    left = tf.pad(x[:, :, 1:, :], [[0, 0], [0, 0], [0, 1], [0, 0]])
    right = tf.pad(x[:, :, :-1, :], [[0, 0], [0, 0], [1, 0], [0, 0]])

    weights = tf.cast(
        tf.reshape(self.neighbour_weights, (4, 1, 1, 1, self.channels)), x.dtype)
    return (x
            + weights[0] * up
            + weights[1] * down
            + weights[2] * left
            + weights[3] * right)

  def get_config(self):
    return super().get_config()


class DynamicMultiModalFusion(layers.Layer):
  """Fuses modality feature maps with content-dependent attention weights.

  Each modality is pooled to a global descriptor; the concatenated descriptors
  go through a small MLP that emits one softmax weight per modality, so the
  mixing depends on the sample instead of being a fixed learned constant. The
  weighted sum is then refined by two 1x1 convolutions and layer normalised.

  All modalities are given as spatial maps [B, H, W, C] with the same H, W.
  """

  def __init__(self, d_model: int, num_modalities: int = 3,
               hidden_dim: int = 64, dropout_rate: float = 0.1, **kwargs):
    super().__init__(**kwargs)
    self.d_model = d_model
    self.num_modalities = num_modalities
    self.hidden_dim = hidden_dim
    self.dropout_rate = dropout_rate

    self.attention_mlp = tf.keras.Sequential([
        layers.Dense(hidden_dim, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(max(hidden_dim // 2, 1), activation="relu"),
        layers.Dense(num_modalities, activation="softmax"),
    ])
    self.global_pool = layers.GlobalAveragePooling2D()
    self.cross_proj = layers.Conv2D(d_model, 1, activation="swish", padding="same")
    self.output_proj = layers.Conv2D(d_model, 1, activation="relu", padding="same")
    self.modal_norm = layers.LayerNormalization(axis=-1)

  def call(self, modalities, training: bool = False):
    """Args: list of [B, H, W, C] tensors. Returns [B, H, W, d_model]."""
    if len(modalities) != self.num_modalities:
      raise ValueError(
          f"Expected {self.num_modalities} modalities, got {len(modalities)}")

    global_features = [self.global_pool(mod) for mod in modalities]
    attention = self.attention_mlp(
        tf.concat(global_features, axis=-1), training=training)  # [B, M]

    weighted = [
        mod * tf.reshape(attention[:, i], [-1, 1, 1, 1])
        for i, mod in enumerate(modalities)
    ]
    fused = tf.add_n(weighted)

    enhanced = self.cross_proj(fused)
    enhanced = self.output_proj(enhanced)
    return self.modal_norm(enhanced)

  def get_config(self):
    config = super().get_config()
    config.update({
        "d_model": self.d_model,
        "num_modalities": self.num_modalities,
        "hidden_dim": self.hidden_dim,
        "dropout_rate": self.dropout_rate,
    })
    return config
