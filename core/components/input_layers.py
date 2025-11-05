import keras
import tensorflow as tf

@keras.utils.register_keras_serializable()
class InputPtEtaPhiELayer(keras.layers.Layer):
    """
    Converts particle features from (pt, eta, phi, E)
    to (px, py, pz, E), while passing through residual features.

    A boolean mask can be passed to `call(inputs, mask=...)`
    where True marks valid entries. Masked entries are zeroed 
    before computation to avoid numerical issues.
    
    Expected input shape: (..., N_features) where the first 4 are
    (pt, eta, phi, E), followed by any residual features.
    """

    def __init__(self,log_E,padding_value, **kwargs):
        super().__init__(**kwargs)
        self.log_E = log_E
        self.padding_value = padding_value

    def call(self, inputs, mask=None):
        # Split input features
        pt = inputs[..., 0:1]
        eta = inputs[..., 1:2]
        phi = inputs[..., 2:3]
        energy = inputs[..., 3:4]
        residual = inputs[..., 4:]

        if mask is not None:
            # Ensure mask is broadcastable to feature dimension
            mask = tf.cast(mask, tf.bool)
            mask = tf.expand_dims(mask, axis=-1)

            # Replace masked entries with zeros for stable computation
            safe_pt = tf.where(mask, pt, tf.zeros_like(pt))
            safe_eta = tf.where(mask, eta, tf.zeros_like(eta))
            safe_phi = tf.where(mask, phi, tf.zeros_like(phi))
            safe_energy = tf.where(mask, energy, tf.ones_like(energy))
        else:
            # No mask passed — process everything
            safe_pt, safe_eta, safe_phi, safe_energy = pt, eta, phi, energy
        if self.log_E:
            safe_energy = tf.where(safe_energy > 0, safe_energy, tf.ones_like(safe_energy) * 1e-6)
            safe_energy = tf.math.log(safe_energy)

        # Compute Cartesian components
        px = safe_pt * tf.cos(safe_phi)
        py = safe_pt * tf.sin(safe_phi)
        pz = safe_pt * tf.sinh(tf.clip_by_value(safe_eta, -10.0, 10.0))

        # Concatenate outputs
        outputs = tf.concat([px, py, pz, safe_energy, residual], axis=-1)

        # If mask exists, restore masked entries to 0 (or optionally NaN / -999)
        if mask is not None:
            outputs = tf.where(mask, outputs, tf.ones_like(outputs) * self.padding_value)

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "log_E": self.log_E,
            "padding_value": self.padding_value,
        })
        return config

@keras.utils.register_keras_serializable()
class InputMetPhiLayer(keras.layers.Layer):
    """
    Converts MET representation from (met, met_phi)
    to (met_x, met_y).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        tf.debugging.assert_equal(
            tf.shape(inputs)[-1], 2,
            message="Input tensor must have exactly 2 features: (met, met_phi)"
        )

        met = inputs[..., 0:1]
        met_phi = inputs[..., 1:2]

        met_x = met * tf.cos(met_phi)
        met_y = met * tf.sin(met_phi)

        return tf.concat([met_x/ 1e3, met_y/ 1e3], axis=-1)

    def get_config(self):
        config = super().get_config()
        return config