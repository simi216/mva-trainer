import tensorflow as tf
import keras

@keras.utils.register_keras_serializable()
class AssignmentLoss(keras.losses.Loss):
    """
    A custom loss function for assignment problems, combining cross-entropy loss 
    with a soft exclusion penalty to enforce constraints on the sum of probabilities.
    Attributes:
        lambda_excl (float): Weight for the exclusion penalty term. Default is 0.0.
        name (str): Name of the loss function. Default is "assignment_loss".
    Methods:
        call(y_true, y_pred, sample_weight=None):
            Computes the total loss as the sum of cross-entropy loss and the 
            weighted exclusion penalty.
    Args:
        lambda_excl (float, optional): Weight for the exclusion penalty term. 
            Controls the strength of the penalty for violating the sum of probabilities constraint. 
            Default is 0.0.
        name (str, optional): Name of the loss function. Default is "assignment_loss".
        **kwargs: Additional keyword arguments passed to the base class.
    Call Args:
        y_true (Tensor): Ground truth tensor of shape (batch_size, num_jets, 2), 
            where the second dimension represents one-hot encoded targets.
        y_pred (Tensor): Predicted tensor of shape (batch_size, num_jets, 2,), 
            where the second dimension represents softmax outputs.
        sample_weight (Tensor, optional): Optional tensor of shape (batch_size,) 
            representing weights for each sample in the batch.
        Tensor: A tensor of shape (batch_size,) representing the per-sample loss.
    """
    def __init__(self, lambda_excl=0.0, name="assignment_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.lambda_excl = lambda_excl

    def call(self, y_true, y_pred, sample_weight=None):
        """
        Computes the total loss as the sum of cross-entropy loss and the 
        weighted exclusion penalty.
        Args:
            y_true (Tensor): Ground truth tensor of shape (batch_size, num_jets, 2), 
                where the second dimension represents one-hot encoded targets.
            y_pred (Tensor): Predicted tensor of shape (batch_size, num_jets, 2,), 
                where the second dimension represents softmax outputs.
            sample_weight (Tensor, optional): Optional tensor of shape (batch_size,) 
                representing weights for each sample in the batch.
        Returns:
            Tensor of shape (batch_size,) — per-sample loss
        """
        # Cross-entropy loss per lepton, summed over lepton dim
        cross_entropy = keras.losses.categorical_crossentropy(y_true, y_pred, axis=1)
        ce_loss = tf.reduce_mean(cross_entropy, axis=-1)  # shape: (batch_size,)

        # Exclusion penalty: penalize if sum of probabilities over jets deviates from 1
        sum_probs = tf.reduce_sum(y_pred, axis=-1)  # shape: (batch_size, num_jets)
        exclusion_penalty = tf.reduce_sum((tf.nn.relu(sum_probs - 1.0)) ** 2, axis=-1)  # shape: (batch_size,)
        excl_loss = self.lambda_excl * exclusion_penalty  # shape: (batch_size,)
        total_loss = ce_loss + excl_loss  # shape: (batch_size,)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, total_loss.dtype)
            total_loss = total_loss * sample_weight
        return total_loss
