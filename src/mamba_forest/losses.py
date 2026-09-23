"""Training objective for class-imbalanced forest-type segmentation."""

from __future__ import annotations

import tensorflow as tf

# Per-class weights of the cross-entropy term. Planted forest is the hardest
# and rarest of the three forest types and carries the largest weight.
DEFAULT_CLASS_WEIGHTS = (1.0, 1.0, 3.0, 1.0, 1.0, 1.2, 1.0, 2.0, 1.0)


class ComboLoss(tf.keras.losses.Loss):
  """Convex combination of weighted cross-entropy and soft Dice loss.

      L = alpha * L_WCCE + (1 - alpha) * L_Dice

  Cross-entropy optimises per-pixel accuracy and is dominated by the frequent
  classes, so it is weighted per class and smoothed; the Dice term is computed
  per class and therefore gives the rare classes a comparable pull on the
  gradient. With the default ``alpha`` of 0.4, 40% of the objective is pixel
  accuracy and 60% region overlap.

  Args:
    num_classes: number of segmentation classes.
    alpha: weight of the cross-entropy term, in [0, 1].
    label_smoothing: label smoothing applied to the cross-entropy term.
    class_weights: one weight per class, or None for uniform weights.
    smooth: numerical stabiliser of the Dice quotient.
  """

  def __init__(self, num_classes: int, alpha: float = 0.4,
               label_smoothing: float = 0.05,
               class_weights=DEFAULT_CLASS_WEIGHTS, smooth: float = 1e-6,
               name: str = "combo_loss", **kwargs):
    super().__init__(name=name, **kwargs)
    if not 0.0 <= alpha <= 1.0:
      raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if class_weights is not None and len(class_weights) != num_classes:
      raise ValueError(
          f"class_weights must have {num_classes} entries, "
          f"got {len(class_weights)}")
    self.num_classes = num_classes
    self.alpha = alpha
    self.smooth = smooth
    self.label_smoothing = label_smoothing
    self.class_weights = tuple(class_weights) if class_weights else None

  def _weighted_cross_entropy(self, y_true, y_pred):
    """Per-pixel cross-entropy, smoothed and weighted by the true class."""
    smoothing = self.label_smoothing
    if smoothing > 0:
      y_smooth = (y_true * (1.0 - smoothing)
                  + smoothing / tf.cast(self.num_classes, y_true.dtype))
    else:
      y_smooth = y_true

    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    per_pixel = -tf.reduce_sum(y_smooth * tf.math.log(y_pred), axis=-1)

    if self.class_weights is not None:
      weights = tf.constant(self.class_weights, dtype=y_pred.dtype)
      # The weight of a pixel is the weight of its ground-truth class; the
      # un-smoothed labels are used so smoothing does not blur the weighting.
      per_pixel *= tf.reduce_sum(y_true * weights, axis=-1)
    return tf.reduce_mean(per_pixel)

  def _dice(self, y_true, y_pred):
    """Soft Dice loss averaged over the classes present in the batch."""
    batch = tf.shape(y_true)[0]
    y_true_flat = tf.reshape(y_true, [batch, -1, self.num_classes])
    y_pred_flat = tf.reshape(y_pred, [batch, -1, self.num_classes])

    intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=1)
    union = (tf.reduce_sum(y_true_flat, axis=1)
             + tf.reduce_sum(y_pred_flat, axis=1))
    dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

    # Average only over the classes that occur: a class that is in neither the
    # labels nor the prediction scores a perfect smooth / smooth = 1, which
    # would reward the model for ignoring the rare classes.
    present = tf.cast(tf.reduce_sum(y_true_flat, axis=1) > 0, dice.dtype)
    return 1.0 - (tf.reduce_sum(dice * present)
                  / tf.maximum(tf.reduce_sum(present), 1.0))

  def call(self, y_true, y_pred):
    y_true = tf.cast(y_true, y_pred.dtype)
    cross_entropy = self._weighted_cross_entropy(y_true, y_pred)
    dice = self._dice(y_true, y_pred)
    return self.alpha * cross_entropy + (1.0 - self.alpha) * dice

  def get_config(self):
    config = super().get_config()
    config.update({
        "num_classes": self.num_classes,
        "alpha": self.alpha,
        "label_smoothing": self.label_smoothing,
        "class_weights": self.class_weights,
        "smooth": self.smooth,
    })
    return config
