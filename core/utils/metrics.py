import keras
import tensorflow as tf


@keras.utils.register_keras_serializable()
class AssignmentAccuracy(keras.metrics.Metric):
    def __init__(self, name="assignment_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.argmax(y_true, axis=-2), tf.int32) # shape: (batch_size, 2)
        y_pred = tf.cast(tf.argmax(y_pred, axis=-2), tf.int32) # shape: (batch_size, 2)
        matches = tf.reduce_all(tf.equal(y_true, y_pred), axis=-1) # shape: (batch_size,)
        matches = tf.cast(matches, self.dtype) # shape: (batch_size,)
        count = tf.shape(y_true)[0]
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            matches = matches * sample_weight
            count = tf.reduce_sum(sample_weight)
        self.total.assign_add(tf.reduce_sum(matches))
        self.count.assign_add(tf.cast(count, self.dtype))

    def result(self):
        return self.total / self.count

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


@keras.utils.register_keras_serializable()
class RelativeRegressionDeviation(keras.losses.Loss):
    def __init__(
        self,
        mode="component",   # "component" or "magnitude" or "log"
        alpha=1.0,          # floor for denominator (in same units as momenta)
        var_weights=None,   # shape (num_vars,) or None
        name="relative_regression_deviation",
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
