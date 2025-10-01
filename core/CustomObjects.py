import numpy as np
"""
CustomObjects Module
====================
This module contains custom Keras layers, loss functions, and utility functions 
designed for specialized machine learning tasks. These components are implemented 
to extend the functionality of Keras and TensorFlow, providing additional tools 
for model development and training.
Contents
--------
1. **Custom Loss Functions**:
    - `accuracy`: Computes the accuracy of predictions for one-hot encoded tensors.
    - `CombinedLoss`: Combines assignment and regression losses for multi-output models.
    - `RegressionLoss`: Custom regression loss function using normalized mean squared error.
    - `AssignmentLoss`: Custom loss function for assignment tasks with cross-entropy and exclusion penalty.
2. **Custom Layers**:
    - `DataNormalizationLayer`: Normalizes input data using precomputed mean and standard deviation.
    - `SplitLayer`: Splits input tensors along a specified axis.
    - `PrependLearnedVector`: Prepends a learned vector to the input tensor along the sequence axis.
    - `AppendTrueMask`: Appends a specified number of `True` values to the input tensor.
    - `JetMaskingLayer`: Generates a mask for input tensors based on a specified padding value.
    - `TemporalSoftmax`: Applies a softmax operation along a specified axis with optional masking support.
3. **Utilities**:
    - Serialization support for all custom components using `keras.utils.register_keras_serializable`.
Each component is designed to be modular and reusable, with detailed documentation 
provided for their attributes, methods, and usage examples.


Notes:
All custom objects are registered with Keras using the `@keras.utils.register_keras_serializable()` decorator.
When a model is loaded using the utilities provided by the BaseModel classes, these custom objects will be automatically recognized and used.

"""
import pandas as pd

import tensorflow as tf
import keras
from keras.src.api_export import keras_export
from keras.src import activations
from keras.src import backend
import sklearn as sk
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.losses import loss as loss_module
from keras.src import tree
import warnings



@keras.utils.register_keras_serializable()
def accuracy(y_true, y_pred):
    """
    Computes the accuracy of predictions.
    This function calculates the accuracy by comparing the predicted class indices
    with the true class indices. It assumes that both `y_true` and `y_pred` are
    one-hot encoded tensors.
    Args:
        y_true (tf.Tensor): A tensor of true labels in one-hot encoded format.
        y_pred (tf.Tensor): A tensor of predicted probabilities or logits.
    Returns:
        tf.Tensor: A scalar tensor representing the mean accuracy of the predictions.
    """
    

    # --- Calculate accuracy ---
    correct_predictions = tf.equal(
        tf.argmax(y_pred, axis=1), tf.argmax(y_true, axis=1)
    )
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return accuracy


@keras.utils.register_keras_serializable()
class CombinedLoss(keras.losses.Loss):
    """
    A custom loss function that combines an assignment loss and a regression loss.
    This loss function is designed to handle multi-output models where the outputs
    are divided into assignment and regression tasks. The total loss is computed
    as the sum of the assignment loss and a weighted regression loss.
    Attributes:
        assignment_weigth (float): Weight applied to the regression loss when combining
            it with the assignment loss.
        assignment_loss (keras.losses.Loss): Instance of the AssignmentLoss class used
            to compute the assignment loss.
        regression_loss (keras.losses.Loss): Instance of the RegressionLoss class used
            to compute the regression loss.
    Args:
        name (str): Name of the loss function. Defaults to "combined_loss".
        lambda_excl (float): Parameter passed to the AssignmentLoss to control
            exclusion behavior. Defaults to 0.
        assignment_weigth (float): Weight applied to the regression loss. Defaults to 1.
        **kwargs: Additional keyword arguments passed to the base Loss class.
    Methods:
        call(y_true, y_pred, sample_weight=None):
            Computes the combined loss.
            Args:
                y_true (dict): Dictionary containing the true values for the assignment
                    and regression outputs. Keys are "assignment_output" and
                    "regression_output".
                y_pred (dict): Dictionary containing the predicted values for the
                    assignment and regression outputs. Keys are "assignment_output" and
                    "regression_output".
                sample_weight (dict, optional): Dictionary containing sample weights for
                    the assignment and regression outputs. Keys are "assignment_output"
                    and "regression_output". Defaults to None.
            Returns:
                tf.Tensor: The total combined loss as a scalar tensor.
    """
    def __init__(
        self, name="combined_loss", lambda_excl=0, assignment_weigth=1, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.assignment_weigth = assignment_weigth
        self.assignment_loss = AssignmentLoss(
            lambda_excl=lambda_excl, name="assignment_loss"
        )
        self.regression_loss = RegressionLoss(name="regression_loss")

    def call(self, y_true, y_pred, sample_weight=None):
        """
        y_true: dictionary with keys 'assignment' and 'regression'
        y_pred: dictionary with keys 'assignment' and 'regression'
        """

        # --- Extract assignment and regression targets ---
        y_true_assignment, y_true_regression = (
            y_true["assignment_output"],
            y_true["regression_output"],
        )
        y_pred_assignment, y_pred_regression = (
            y_pred["assignment_output"],
            y_pred["regression_output"],
        )

        # --- Calculate assignment loss ---
        assignment_loss = self.assignment_loss(
            y_true_assignment,
            y_pred_assignment,
            sample_weight=sample_weight["assignment_output"],
        )
        # --- Calculate regression loss ---
        regression_loss = self.regression_loss(
            y_true_regression,
            y_pred_regression,
            sample_weight=sample_weight["regression_output"],
        )
        # --- Combine losses ---
        total_loss = assignment_loss + self.assignment_weigth * regression_loss
        return total_loss


@keras.utils.register_keras_serializable(package="CustomUtility")
class RegressionLoss(keras.losses.Loss):
    """
    A custom regression loss function for Keras models.
    This loss function computes the mean squared error (MSE) between the true 
    and predicted regression targets, normalized by the sum of the true values. 
    It also supports optional sample weighting.
    Attributes:
        name (str): The name of the loss function. Defaults to "regression_loss".
    Methods:
        call(y_true, y_pred, sample_weight=None):
            Computes the per-sample loss.
            Args:
                y_true (Tensor): A tensor of shape (batch_size, num_jets, n_regression_targets) 
                    representing the true regression targets.
                y_pred (Tensor): A tensor of shape (batch_size, num_jets, n_regression_targets) 
                    representing the predicted regression targets.
                sample_weight (Tensor, optional): A tensor of weights to apply to each sample. 
                    Defaults to None.
                Tensor: A tensor of shape (batch_size,) representing the per-sample loss.
    """
    def __init__(self, name="regression_loss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred, sample_weight=None):
        """
        y_true: shape (batch_size, num_jets, n_regression_targets) - regression targets
        y_pred: shape (batch_size, num_jets, n_regression_targets) - regression predictions
        Returns:
            Tensor of shape (batch_size,) — per-sample loss
        """
        mse_loss = keras.losses.mean_squared_error(y_true, y_pred) / tf.reduce_sum(
            y_true, keepdims=True
        )
        if sample_weight is not None:
            # Apply sample weights if provided
            mse_loss *= sample_weight
        return tf.reduce_mean(mse_loss)


@keras.utils.register_keras_serializable(package="CustomUtility")
class AssignmentLoss(keras.losses.Loss):
    """
    Custom loss function for assignment tasks, combining cross-entropy loss 
    and a soft exclusion penalty to enforce constraints on the predictions.
    Attributes:
        lambda_excl (float): Weight for the exclusion penalty term. Default is 0.0.
        name (str): Name of the loss function. Default is "assignment_loss".
    Methods:
        call(y_true, y_pred, sample_weight=None):
            Computes the total loss as the sum of cross-entropy loss and 
            the weighted exclusion penalty.
    Args:
        lambda_excl (float, optional): Weight for the exclusion penalty term. 
            Default is 0.0.
        name (str, optional): Name of the loss function. Default is "assignment_loss".
        **kwargs: Additional keyword arguments passed to the base class.
    """
    def __init__(self, lambda_excl=0.0, name="assignment_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.lambda_excl = lambda_excl

    def call(self, y_true, y_pred, sample_weight=None):
        """
        y_true: shape (batch_size, 2, num_jets) - one-hot encoded targets
        y_pred: shape (batch_size, 2, num_jets) - softmax outputs
        Returns:
            Tensor of shape (batch_size,) — per-sample loss
        """
        # Cross-entropy loss per lepton, summed over lepton dim
        cross_entropy = keras.losses.categorical_crossentropy(y_true, y_pred, axis=1)
        ce_loss = tf.reduce_sum(cross_entropy, axis=-1)  # shape: (batch_size,)

        # Soft exclusion penalty
        jet_probs_sum = tf.reduce_sum(y_pred, axis=-1)  # shape: (batch_size, 2)
        violation = tf.nn.relu(jet_probs_sum - 1.0)
        excl_penalty = tf.reduce_sum(
            tf.square(violation), axis=-1
        )  # shape: (batch_size,)
        if sample_weight is not None:
            # Apply sample weights if provided
            ce_loss *= sample_weight
            excl_penalty *= sample_weight

        total_loss = ce_loss + self.lambda_excl * excl_penalty  # shape: (batch_size,)
        return total_loss


@keras.utils.register_keras_serializable()
class RegressionLoss(keras.losses.Loss):
    """
    A custom regression loss function for Keras models.
    This loss function computes the mean squared error (MSE) between the true 
    and predicted regression targets, normalized by the sum of the true values. 
    It also supports optional sample weighting.
    Attributes:
        name (str): The name of the loss function. Defaults to "regression_loss".
    Methods:
        call(y_true, y_pred, sample_weight=None):
            Computes the per-sample loss.
            Args:
                y_true (Tensor): A tensor of shape (batch_size, num_jets, n_regression_targets) 
                    representing the true regression targets.
                y_pred (Tensor): A tensor of shape (batch_size, num_jets, n_regression_targets) 
                    representing the predicted regression targets.
                sample_weight (Tensor, optional): A tensor of weights to apply to the loss 
                    for each sample. Defaults to None.
                Tensor: A tensor of shape (batch_size,) representing the per-sample loss.
    """
    def __init__(self, name="regression_loss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred, sample_weight=None):
        """
        y_true: shape (batch_size, num_jets, n_regression_targets) - regression targets
        y_pred: shape (batch_size, num_jets, n_regression_targets) - regression predictions
        Returns:
            Tensor of shape (batch_size,) — per-sample loss
        """
        mse_loss = keras.losses.mean_squared_error(y_true, y_pred) / tf.reduce_sum(
            y_true, keepdims=True
        )
        if sample_weight is not None:
            # Apply sample weights if provided
            mse_loss *= sample_weight
        return tf.reduce_mean(mse_loss)


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
        y_true (Tensor): Ground truth tensor of shape (batch_size, 2, num_jets), 
            where the second dimension represents one-hot encoded targets.
        y_pred (Tensor): Predicted tensor of shape (batch_size, 2, num_jets), 
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
        y_true: shape (batch_size, 2, num_jets) - one-hot encoded targets
        y_pred: shape (batch_size, 2, num_jets) - softmax outputs
        Returns:
            Tensor of shape (batch_size,) — per-sample loss
        """
        # Cross-entropy loss per lepton, summed over lepton dim
        cross_entropy = keras.losses.categorical_crossentropy(y_true, y_pred, axis=1)
        ce_loss = tf.reduce_sum(cross_entropy, axis=-1)  # shape: (batch_size,)

        # Soft exclusion penalty
        jet_probs_sum = tf.reduce_sum(y_pred, axis=-1)  # shape: (batch_size, 2)
        violation = tf.nn.relu(jet_probs_sum - 1.0)
        excl_penalty = tf.reduce_sum(
            tf.square(violation), axis=-1
        )  # shape: (batch_size,)
        if sample_weight is not None:
            # Apply sample weights if provided
            ce_loss *= sample_weight
            excl_penalty *= sample_weight

        total_loss = ce_loss + self.lambda_excl * excl_penalty  # shape: (batch_size,)
        return total_loss



@keras.utils.register_keras_serializable()
class DataNormalizationLayer(keras.layers.Layer):
    """
    A custom Keras layer for normalizing input data using precomputed mean and standard deviation.
    This layer normalizes the input data along a specified axis by subtracting the mean and dividing
    by the standard deviation. The mean and standard deviation are computed from the provided data
    during initialization.
    Attributes:
        axis (int): The axis along which normalization is performed.
        mean (tf.Tensor): The mean of the data along the specified axis.
        std (tf.Tensor): The standard deviation of the data along the specified axis.
    Methods:
        call(inputs):
            Applies normalization to the input tensor.
        get_config():
            Returns the configuration of the layer as a dictionary.
        from_config(config):
            Creates a layer instance from a configuration dictionary.
    Args:
        data (array-like): The data used to compute the mean and standard deviation.
        axis (int, optional): The axis along which normalization is performed. Defaults to -1.
        **kwargs: Additional keyword arguments for the base `keras.layers.Layer` class.
    Example:
        ```python
        # Example data
        data = np.random.rand(100, 10)
        # Create the normalization layer
        norm_layer = DataNormalizationLayer(data=data, axis=1)
        # Example input
        inputs = tf.random.normal((5, 10))
        # Apply normalization
        normalized_inputs = norm_layer(inputs)
        ```
    """

    def __init__(self, data, axis=-1, **kwargs):
        super().__init__(**kwargs)
        if axis < 0:
            axis += len(data.shape)
        self.axis = axis
        dtype = keras.backend.floatx()
        sum_axis = tuple(i for i in range(len(data.shape)) if i != axis)
        data = tf.convert_to_tensor(data, dtype=dtype)
        self.mean = tf.constant(tf.reduce_mean(data, axis=sum_axis, keepdims=True),dtype=dtype)
        self.std = tf.constant(tf.math.reduce_std(data, axis=sum_axis, keepdims=True), dtype=dtype)

    def call(self, inputs):
        return (inputs - self.mean) / (self.std + 1e-7)

    def get_config(self):
        return {
            **super().get_config(),
            "axis": self.axis,
            "mean": self.mean.numpy().tolist(),
            "std": self.std.numpy().tolist(),
        }

@keras.utils.register_keras_serializable()
class SplitLayer(keras.layers.Layer):
    def __init__(self, split_size, axis=-1, **kwargs):
        super().__init__(**kwargs)  # Split along the second dimension
        self.split_size = split_size
        self.axis = axis  # Axis along which to split
        self.supports_masking = True  # Enable masking support

    def call(self, inputs):
        """
        inputs: shape (batch_size, total_size)
        Returns a list of tensors each of shape (batch_size, split_size)
        """
        return tf.split(inputs, num_or_size_splits=self.split_size, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"split_size": self.split_size})
        config.update({"axis": self.axis})
        return config


@keras.utils.register_keras_serializable()
class PrependLearnedVector(keras.layers.Layer):
    """
    A custom Keras layer that prepends a learned vector to the input tensor along the sequence axis.
    This layer is useful in scenarios where a trainable embedding needs to be prepended to the input
    sequence, such as adding a special token embedding in transformer-based models.
    Attributes:
        embed_dim (int): The dimensionality of the learned vector.
    Methods:
        build(input_shape):
            Creates the trainable weight for the learned vector.
        call(inputs):
            Prepends the learned vector to the input tensor along the sequence axis.
        get_config():
            Returns the configuration of the layer for serialization.
    Args:
        embed_dim (int): The dimensionality of the learned vector.
        **kwargs: Additional keyword arguments for the base Keras Layer class.
    """
    
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        # Create a trainable weight for the learned vector
        self.learned_vector = self.add_weight(
            shape=(1, 1, self.embed_dim),  # (batch, seq, embed)
            initializer='glorot_uniform',
            trainable=True,
            name='learned_vector'
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # Repeat learned vector for each batch entry
        vector = tf.tile(self.learned_vector, [batch_size, 1, 1])
        return tf.concat([vector, inputs], axis=1)  # Prepend along sequence axis

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim})
        return config


@keras.utils.register_keras_serializable()
class AppendTrueMask(tf.keras.layers.Layer):
    """
    A custom Keras layer that appends a specified number of `True` values 
    to the input tensor along the second dimension.
    Attributes:
        n_append (int): The number of `True` values to append to the input tensor.
    Methods:
        call(inputs):
            Appends `True` values to the input tensor along the second dimension.
            Args:
                inputs (tf.Tensor): A 2D or 3D tensor of shape 
                                    (batch_size, original_mask_len) or 
                                    (batch_size, original_mask_len, 1).
            Returns:
                tf.Tensor: A tensor with `n_append` `True` values appended along 
                           the second dimension. The output shape is 
                           (batch_size, original_mask_len + n_append, 1).
        compute_output_shape(input_shape):
            Computes the output shape of the layer.
            Args:
                input_shape (tuple): The shape of the input tensor.
            Returns:
                tuple: The shape of the output tensor, which is 
                       (input_shape[0], input_shape[1] + n_append, 1).
    """

    def __init__(self, n_append=3, **kwargs):
        super().__init__(**kwargs)
        self.n_append = n_append

    def call(self, inputs):
        # inputs: (batch_size, original_mask_len)
        batch_size = tf.shape(inputs)[0]
        # Create tensor of shape (batch_size, n_append) filled with True
        if tf.rank(inputs) == 2:
            inputs = tf.expand_dims(inputs, -1)
        append = tf.ones((batch_size, self.n_append, 1), dtype=tf.bool)
        return tf.concat([inputs, append], axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + self.n_append, 1)


@keras.utils.register_keras_serializable()
class JetMaskingLayer(keras.layers.Layer):
    """
    A custom Keras layer that generates a mask for input tensors based on a specified padding value.
    This layer checks whether the elements in the input tensor are not equal to the given padding value
    and returns a boolean mask. The mask indicates which elements are valid (not equal to the padding value).
    Attributes:
        padding_value (float): The value used for padding in the input tensor. Elements equal to this value
                               will be masked out.
    Methods:
        call(inputs):
            Computes the boolean mask for the input tensor by checking if elements are not equal to the padding value.
            Args:
                inputs (tf.Tensor): The input tensor to be masked.
            Returns:
                tf.Tensor: A boolean tensor of the same shape as the input, with `True` for valid elements
                           and `False` for padded elements.
        get_config():
            Returns the configuration of the layer for serialization.
            Returns:
                dict: A dictionary containing the configuration of the layer, including the padding value.
    """

    def __init__(self, padding_value, **kwargs):
        super().__init__(**kwargs)
        self.padding_value = padding_value

    def call(self, inputs):
        return tf.reduce_all(
            tf.not_equal(inputs, self.padding_value), axis=-1, keepdims=True
        )

    def get_config(self):
        config = super().get_config()
        config.update({"padding_value": self.padding_value})
        return config


@keras.utils.register_keras_serializable()
class TemporalSoftmax(keras.layers.Layer):
    """
    A custom Keras layer that applies a softmax operation along a specified axis,
    with optional masking support. This layer is useful for temporal data where
    masking is required to ignore certain time steps during the softmax computation.
    Attributes:
        axis (int): The axis along which the softmax operation is applied. Defaults to -2.
        supports_masking (bool): Indicates that this layer supports masking.
    Methods:
        call(inputs, mask=None):
            Applies the softmax operation to the inputs along the specified axis.
            If a mask is provided, it is used to ignore certain elements during
            the computation.
        compute_output_shape(input_shape):
            Computes and returns the output shape of the layer, which is the same
            as the input shape.
        get_config():
            Returns the configuration of the layer as a dictionary, including the
            axis attribute.
    Args:
        axis (int, optional): The axis along which to apply the softmax. Defaults to -2.
        **kwargs: Additional keyword arguments for the base Keras Layer class.
    Example:
        ```python
        # Example usage of TemporalSoftmax
        temporal_softmax = TemporalSoftmax(axis=-1)
        inputs = tf.random.uniform((2, 3, 4))
        mask = tf.constant([[1, 1, 0], [1, 0, 0]], dtype=tf.float32)
        outputs = temporal_softmax(inputs, mask=mask)
        print(outputs)
        ```
    """

    def __init__(self, axis=-2, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = tf.cast(mask, inputs.dtype)
            mask = tf.broadcast_to(mask, tf.shape(inputs))
            not_mask = 1.0 - mask
            inputs += not_mask * -1e9
        softmax_output = tf.nn.softmax(inputs, axis=self.axis)
        if mask is not None:
            softmax_output *= mask
        return softmax_output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config