import keras
import tensorflow as tf


class GenerateMask(keras.layers.Layer):
    def __init__(self, padding_value=-999, **kwargs):
        super().__init__(**kwargs)
        self.padding_value = padding_value

    def call(self, inputs):
        # For shape (batch, seq_len, dim), we reduce across dim to get (batch, seq_len, 1)
        not_pad = tf.not_equal(inputs, self.padding_value)
        mask = tf.reduce_any(not_pad, axis=-1)
        return tf.cast(mask, tf.bool)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)

    def get_config(self):
        config = super().get_config()
        config.update({"padding_value": self.padding_value})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
