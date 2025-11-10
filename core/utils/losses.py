import tensorflow as tf
import keras


@keras.utils.register_keras_serializable()
class AssignmentLoss(keras.losses.Loss):
    def __init__(self, lambda_excl=0.0, epsilon=1e-7, name="assignment_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.lambda_excl = lambda_excl
        self.epsilon = epsilon

    def call(self, y_true, y_pred, sample_weight=None):
        # Detect mask: jets where probabilities sum to ~0
        sum_probs = tf.reduce_sum(y_pred, axis=-1)  # (batch, jets)
        mask = tf.cast(sum_probs > self.epsilon, y_pred.dtype)  # (batch, jets)
        mask_expanded = mask[..., tf.newaxis]  # (batch, jets, 1)

        # Number of true leptons (should be 2)
        num_true = tf.reduce_sum(y_true * mask_expanded)
        num_true = tf.maximum(num_true, 1.0)

        # Apply mask and clip predictions
        y_true_masked = y_true * mask_expanded
        y_pred_safe = tf.clip_by_value(y_pred, self.epsilon, 1.0)

        # Cross-entropy (per jet, per lepton)
        ce = -y_true_masked * tf.math.log(y_pred_safe)
        ce_total = tf.reduce_sum(ce, axis=[1, 2])
        ce_loss = ce_total / num_true

        # Exclusion penalty (soft exclusivity)
        if self.lambda_excl > 0:
            penalty_per_jet = tf.math.softplus(sum_probs - 1.0) ** 2
            exclusion_penalty = tf.reduce_sum(penalty_per_jet * mask, axis=-1)
            exclusion_penalty = exclusion_penalty / tf.maximum(tf.reduce_sum(mask, axis=-1), 1.0)
            excl_loss = self.lambda_excl * exclusion_penalty
        else:
            excl_loss = 0.0

        total_loss = ce_loss + excl_loss

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, total_loss.dtype)
            sample_weight = tf.reshape(sample_weight, [-1])
            total_loss = total_loss * sample_weight

        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update({"lambda_excl": self.lambda_excl, "epsilon": self.epsilon})
        return config

import tensorflow as tf
from tensorflow import keras

@keras.utils.register_keras_serializable()
class RegressionLoss(keras.losses.Loss):
    def __init__(
        self,
        mode="component",   # "component" or "magnitude" or "log"
        alpha=1.0,          # floor for denominator (in same units as momenta)
        var_weights=None,   # shape (num_vars,) or None
        name="regression_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        assert mode in ("component", "magnitude")
        self.mode = mode
        self.alpha = float(alpha)
        self.var_weights = (
            tf.constant(var_weights, dtype=tf.float32) if var_weights is not None else None
        )

    def call(self, y_true, y_pred, sample_weight=None):
        """
        y_true, y_pred: shape (batch, n_items, n_vars)
        """

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if self.mode == "component":
            # denom = max(alpha, |y_true|) per-component
            denom = tf.maximum(tf.abs(y_true), self.alpha)
            rel = (y_true - y_pred) / denom  # relative error per component
            sq = tf.square(rel)  # (batch, n_items, n_vars)

        elif self.mode == "magnitude":
            # compute vector magnitude per item (excluding energy if present)
            # assume last axis order: (px, py, pz, [E])
            vec = y_true[..., :3]
            vec_pred = y_pred[..., :3]
            mag = tf.norm(vec, axis=-1, keepdims=True)  # (batch, n_items, 1)
            mag_pred = tf.norm(vec_pred, axis=-1, keepdims=True)
            denom = tf.maximum(mag, self.alpha)  # (batch, n_items, 1)
            rel = (mag - mag_pred) / denom  # relative error on magnitude
            sq = tf.square(rel)  # (batch, n_items, 1)
            # If you want to include other vars (like E), append their component-wise relative errors:
            if y_true.shape[-1] > 3:
                extra_true = y_true[..., 3:]
                extra_pred = y_pred[..., 3:]
                denom_extra = tf.maximum(tf.abs(extra_true), self.alpha)
                rel_extra = (extra_true - extra_pred) / denom_extra
                sq_extra = tf.square(rel_extra)
                # concatenate along last axis
                sq = tf.concat([sq, sq_extra], axis=-1)

        # apply per-variable weights if given
        if self.var_weights is not None:
            # Ensure shape broadcastable: (n_vars,) or (n_vars_of_sq)
            sq = sq * self.var_weights

        # reduce: mean over vars and items, produce per-sample loss
        per_sample = tf.reduce_mean(sq, axis=[1, 2])  # (batch,)

        # apply sample weights if provided
        if sample_weight is not None:
            sample_weight = tf.reshape(tf.cast(sample_weight, per_sample.dtype), [-1])
            per_sample = per_sample * sample_weight

        return per_sample

    def get_config(self):
        config = super().get_config()
        config.update({
            "mode": self.mode,
            "alpha": self.alpha,
            "var_weights": None if self.var_weights is None else self.var_weights.numpy().tolist(),
        })
        return config
