"""Losses for class-imbalanced segmentation."""

from __future__ import annotations

import tensorflow as tf


class ComboLoss(tf.keras.losses.Loss):
  """Convex combination of categorical cross-entropy and soft Dice loss.

  Cross-entropy optimises per-pixel accuracy and is dominated by the frequent
  classes; the Dice term is computed per class and therefore gives the rare
  classes (planted forest, tree crops) a comparable weight. ``alpha`` mixes the
  two: ``alpha * CE + (1 - alpha) * Dice``.

  Args:
    num_classes: number of segmentation classes.
    alpha: weight of the cross-entropy term, in [0, 1].
    label_smoothing: label smoothing for the cross-entropy term.
    smooth: numerical stabiliser of the Dice quotient.
  """

  def __init__(self, num_classes: int, alpha: float = 0.5,
               label_smoothing: float = 0.1, smooth: float = 1e-6,
               name: str = "combo_loss", **kwargs):
    super().__init__(name=name, **kwargs)
    if not 0.0 <= alpha <= 1.0:
      raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    self.num_classes = num_classes
    self.alpha = alpha
    self.smooth = smooth
    self.label_smoothing = label_smoothing
    self.cce = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=label_smoothing)

  def call(self, y_true, y_pred):
    cce = self.cce(y_true, y_pred)

    # Flatten the spatial dimensions: [B, H, W, C] -> [B, H * W, C].
    y_true_flat = tf.reshape(y_true, [tf.shape(y_true)[0], -1, self.num_classes])
    y_pred_flat = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1, self.num_classes])

    intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=1)
    union = (tf.reduce_sum(y_true_flat, axis=1)
             + tf.reduce_sum(y_pred_flat, axis=1))
    dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

    # Only average over classes that occur in the batch: a class that is
    # neither in the labels nor in the prediction gets a perfect dice of
    # smooth / smooth = 1, which would reward ignoring the rare classes.
    present = tf.cast(tf.reduce_sum(y_true_flat, axis=1) > 0, dice.dtype)
    dice_loss = 1.0 - (tf.reduce_sum(dice * present)
                       / tf.maximum(tf.reduce_sum(present), 1.0))

    return self.alpha * cce + (1.0 - self.alpha) * dice_loss

  def get_config(self):
    config = super().get_config()
    config.update({
        "num_classes": self.num_classes,
        "alpha": self.alpha,
        "label_smoothing": self.label_smoothing,
        "smooth": self.smooth,
    })
    return config
