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