import keras
import tensorflow as tf

class OutputUpScaleLayer(keras.layers.Layer):
    def __init__(self, name="OutputUpScaleLayer"):
        super().__init__(name=name)

    def build(self, input_shape):
        self.std = self.add_weight(
            shape=input_shape[1:],
            initializer="ones",
            trainable=False,
            name="output_std",
        )
        self.mean = self.add_weight(
            shape=input_shape[1:],
            initializer="zeros",
            trainable=False,
            name="output_mean",
        )

    def call(self, inputs):
        return inputs * self.std + self.mean

    def adapt_normalization(self, data):
        data = tf.convert_to_tensor(data)
        mean = tf.reduce_mean(data, axis=0)
        std = tf.math.reduce_std(data, axis=0)

        self.mean.assign(mean)
        self.std.assign(std)

    def get_config(self):
        config = super().get_config()
        return config