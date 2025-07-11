import numpy as np
import pandas as pd

import tensorflow as tf
import keras
from keras.src.api_export import keras_export
from keras.src import activations
from keras.src import backend
import sklearn as sk
import onnx
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.losses import loss as loss_module
from keras.src import tree
import warnings



@keras.utils.register_keras_serializable()
def accuracy(y_true, y_pred):
    """
    y_true: shape (batch_size, 2, num_jets) - one-hot encoded targets
    y_pred: shape (batch_size, 2, num_jets) - softmax outputs
    """

    # --- Calculate accuracy ---
    correct_predictions = tf.equal(
        tf.argmax(y_pred, axis=1), tf.argmax(y_true, axis=1)
    )
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return accuracy


@keras.utils.register_keras_serializable()
class CombinedLoss(keras.losses.Loss):
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

    @classmethod
    def from_config(cls, config):
        axis = config.pop("axis")
        mean = tf.constant(config.pop("mean"))
        std = tf.constant(config.pop("std"))
        instance = cls(data=tf.zeros_like(mean), axis=axis, **config)
        instance.mean = mean
        instance.std = std
        return instance

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