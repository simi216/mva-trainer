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
            exclusion_penalty = exclusion_penalty / tf.maximum(
                tf.reduce_sum(mask, axis=-1), 1.0
            )
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


@keras.utils.register_keras_serializable()
class RegressionLoss(keras.losses.Loss):
    def __init__(
        self,
        alpha=1.0,  # floor for denominator (in same units as momenta)
        var_weights=None,  # shape (num_vars,) or None
        name="regression_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.alpha = float(alpha)
        self.var_weights = (
            tf.constant(var_weights, dtype=tf.float32)
            if var_weights is not None
            else None
        )

    def call(self, y_true, y_pred, sample_weight=None):
        """
        y_true, y_pred: shape (batch, n_items, n_vars)
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        rel = y_true - y_pred
        sq = tf.square(rel)  # (batch, n_items, n_vars)

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
        config.update(
            {
                "alpha": self.alpha,
                "var_weights": (
                    None
                    if self.var_weights is None
                    else self.var_weights.numpy().tolist()
                ),
            }
        )
        return config


@keras.utils.register_keras_serializable()
class MagnitudeDirectionLoss(keras.losses.Loss):
    """
    Loss for two neutrino 3-momenta:
    y_true, y_pred have shape (batch, 2, 3)

    L = w_r * relative magnitude loss + w_theta * angular loss
    """

    def __init__(
        self,
        alpha=1.0,  # stabilizer for relative magnitude
        w_mag=1.0,  # weight for magnitude term
        w_dir=1.0,  # weight for direction term
        name="magdir_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.alpha = float(alpha)
        self.w_mag = float(w_mag)
        self.w_dir = float(w_dir)

    def call(self, y_true, y_pred, sample_weight=None):

        y_true = tf.cast(y_true, tf.float32)  # (B,2,3)
        y_pred = tf.cast(y_pred, tf.float32)  # (B,2,3)

        eps = 1e-8

        # --- magnitudes ---
        r_true = tf.norm(y_true, axis=-1)  # (B,2)
        r_pred = tf.norm(y_pred, axis=-1)  # (B,2)

        mag_diff = r_true - r_pred
        mag_loss = tf.square(mag_diff) / (tf.square(r_true) + self.alpha**2 + eps)

        # --- directions ---
        # normalize
        t_norm = y_true / (tf.expand_dims(r_true, -1) + eps)
        p_norm = y_pred / (tf.expand_dims(r_pred, -1) + eps)

        # cosine similarity
        cos_theta = tf.reduce_sum(t_norm * p_norm, axis=-1)  # (B,2)
        cos_theta = tf.clip_by_value(cos_theta, -1.0, 1.0)

        # angular loss (0 if same direction)
        dir_loss = 1.0 - cos_theta

        # --- combine ---
        per_item = self.w_mag * mag_loss + self.w_dir * dir_loss  # (B,2)

        # mean over the two neutrinos
        per_sample = tf.reduce_mean(per_item, axis=1)  # (B,)

        # optional sample weights
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, per_sample.dtype)
            per_sample *= tf.reshape(sample_weight, [-1])

        return per_sample

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "alpha": self.alpha,
                "w_mag": self.w_mag,
                "w_dir": self.w_dir,
            }
        )
        return cfg
