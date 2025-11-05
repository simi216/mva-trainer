import tensorflow as tf
import keras


@keras.utils.register_keras_serializable()
class AssignmentLoss(keras.losses.Loss):
    def __init__(self, lambda_excl=0.0, epsilon=1e-7, name="assignment_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.lambda_excl = lambda_excl
        self.epsilon = epsilon

    def call(self, y_true, y_pred, sample_weight=None):
        """
        Computes assignment loss with proper masking and sample weighting.

        Args:
            y_true: (batch_size, num_jets, 2) - one-hot encoded targets
            y_pred: (batch_size, num_jets, 2) - predicted probabilities
            sample_weight: (batch_size,) - optional per-sample weights

        Returns:
            loss: (batch_size,) - per-sample loss
        """
        # Detect mask: jets where probabilities sum to ~0 are masked
        sum_probs = tf.reduce_sum(y_pred, axis=-1)  # (batch, jets)
        mask = tf.cast(sum_probs > self.epsilon, y_pred.dtype)  # (batch, jets)
        mask_expanded = mask[..., tf.newaxis]  # (batch, jets, 1)

        # Count valid jets per sample for normalization
        num_valid = tf.maximum(tf.reduce_sum(mask, axis=-1), 1.0)  # (batch,)

        # ============ Cross-Entropy Loss ============
        # Ensure y_true is also masked (safety)
        y_true_masked = y_true * mask_expanded

        # Clip predictions to prevent log(0)
        y_pred_safe = tf.clip_by_value(y_pred, self.epsilon, 1.0)

        # Replace masked positions with dummy value (won't affect loss since y_true is 0 there)
        y_pred_masked = tf.where(
            mask_expanded > 0,
            y_pred_safe,
            tf.ones_like(y_pred_safe),  # log(1) = 0, so no contribution
        )

        # Compute cross-entropy
        ce = -y_true_masked * tf.math.log(y_pred_masked)  # (batch, jets, 2)

        # Sum over jets and leptons, then normalize by number of valid jets
        ce_total = tf.reduce_sum(ce, axis=[1, 2])  # (batch,)
        ce_loss = ce_total / num_valid  # (batch,)

        # ============ Exclusion Penalty ============
        if self.lambda_excl > 0:
            # Penalize when sum of probabilities > 1 (jet assigned to both leptons)
            penalty_per_jet = tf.nn.relu(sum_probs - 1.0) ** 2  # (batch, jets)

            # Apply mask: only penalize valid jets
            exclusion_penalty = tf.reduce_sum(
                penalty_per_jet * mask, axis=-1
            )  # (batch,)

            # Normalize by number of valid jets
            exclusion_penalty = exclusion_penalty / num_valid  # (batch,)

            excl_loss = self.lambda_excl * exclusion_penalty
        else:
            excl_loss = 0.0

        # ============ Total Loss ============
        total_loss = ce_loss + excl_loss  # (batch,)

        # ============ Apply Sample Weights ============
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, total_loss.dtype)
            # Ensure sample_weight has correct shape
            sample_weight = tf.reshape(sample_weight, [-1])  # (batch,)
            total_loss = total_loss * sample_weight  # (batch,)

        return total_loss

    def get_config(self):
        config = super().get_config()
        config.update({"lambda_excl": self.lambda_excl, "epsilon": self.epsilon})
        return config


@keras.utils.register_keras_serializable()
class RegressionLoss(keras.losses.Loss):
    def __init__(self, name="regression_loss", **kwargs):
        super().__init__(name=name, **kwargs)
    
    def call(self, y_true, y_pred, sample_weight=None):
        """
        Computes Relative Square Error (RAE) regression loss with sample weighting.

        Args:
            y_true: (batch_size, num_leptons, num_regression_vars) - true regression targets
            y_pred: (batch_size, num_leptons, num_regression_vars) - predicted regression outputs
            sample_weight: (batch_size,) - optional per-sample weights

        Returns:
            loss: (batch_size,) - per-sample loss
        """
        # ============ Relative Square Error ============
        error = tf.square(y_true - y_pred)  # (batch, NUM_LEPTONS, regression_variables)

        std = tf.math.reduce_std(y_true, axis=0)  # (NUM_LEPTONS, regression_variables)
        std = tf.where(std < 1e-6, tf.ones_like(std), std)  # Prevent division by zero
        error /= tf.square(std)  # Normalize by variance

        # Mean over leptons and regression variables
        error_mean = tf.reduce_mean(error, axis=[1, 2])  # (batch,)

        # ============ Apply Sample Weights ============
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, error_mean.dtype)
            # Ensure sample_weight has correct shape
            sample_weight = tf.reshape(sample_weight, [-1])  # (batch,)
            error_mean = error_mean * sample_weight  # (batch,)

        return error_mean