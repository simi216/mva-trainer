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

    def __init__(self, log_E, padding_value, **kwargs):
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
            safe_energy = tf.where(
                safe_energy > 0, safe_energy, tf.ones_like(safe_energy) * 1e-6
            )
            safe_energy = tf.math.log(safe_energy)

        # Compute Cartesian components
        px = safe_pt * tf.cos(safe_phi)
        py = safe_pt * tf.sin(safe_phi)
        pz = safe_pt * tf.sinh(tf.clip_by_value(safe_eta, -10.0, 10.0))

        # Concatenate outputs
        outputs = tf.concat([px, py, pz, safe_energy, residual], axis=-1)

        # If mask exists, restore masked entries to 0 (or optionally NaN / -999)
        if mask is not None:
            outputs = tf.where(
                mask, outputs, tf.ones_like(outputs) * self.padding_value
            )

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "log_E": self.log_E,
                "padding_value": self.padding_value,
            }
        )
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
            tf.shape(inputs)[-1],
            2,
            message="Input tensor must have exactly 2 features: (met, met_phi)",
        )

        met = inputs[..., 0:1]
        met_phi = inputs[..., 1:2]

        met_x = met * tf.cos(met_phi)
        met_y = met * tf.sin(met_phi)

        return tf.concat([met_x / 1e3, met_y / 1e3], axis=-1)

    def get_config(self):
        config = super().get_config()
        return config


@keras.utils.register_keras_serializable()
class ComputeHighLevelFeatures(keras.layers.Layer):
    """
    Computes high-level feature relational features between jets and leptons.
    """

    def __init__(self, padding_value, **kwargs):
        super().__init__(**kwargs)
        self.padding_value = padding_value

    def call(self, jet_input, lepton_input, jet_mask=None, lepton_mask=None):
        # Split jet features
        jet_input = tf.expand_dims(
            jet_input, axis=2
        )  # Shape: (batch_size, n_jets, 1, n_features)
        lepton_input = tf.expand_dims(
            lepton_input, axis=1
        )  # Shape: (batch_size, 1, n_leptons, n_features)

        jet_lepton_mask = None
        if jet_mask is not None:
            jet_mask = tf.expand_dims(
                jet_mask, axis=2
            )  # Shape: (batch_size, n_jets, 1)
            jet_lepton_mask = tf.cast(jet_mask, tf.bool)
        if lepton_mask is not None:
            lepton_mask = tf.expand_dims(
                lepton_mask, axis=1
            )  # Shape: (batch_size, 1, n_leptons)
            if jet_lepton_mask is not None:
                jet_lepton_mask = jet_lepton_mask & lepton_mask
            else:
                jet_lepton_mask = lepton_mask

        jet_px = jet_input[..., 0:1]
        jet_py = jet_input[..., 1:2]
        jet_pz = jet_input[..., 2:3]
        jet_E = jet_input[..., 3:4]

        # Split lepton features
        lep_px = lepton_input[..., 0:1]
        lep_py = lepton_input[..., 1:2]
        lep_pz = lepton_input[..., 2:3]
        lep_E = lepton_input[..., 3:4]

        # Compute delta R
        import math as m
        delta_eta = tf.atanh(
            jet_pz / tf.sqrt(jet_px**2 + jet_py**2 + jet_pz**2)
        ) - tf.atanh(lep_pz / tf.sqrt(lep_px**2 + lep_py**2 + lep_pz**2))
        delta_phi = tf.atan2(jet_py, jet_px) - tf.atan2(lep_py, lep_px)
        delta_phi = tf.math.floormod(
            delta_phi + tf.constant(m.pi), 2 * tf.constant(m.pi)
        ) - tf.constant(m.pi)
        delta_R = tf.sqrt(delta_eta**2 + delta_phi**2)

        # Compute invariant mass
        total_E = jet_E + lep_E
        total_px = jet_px + lep_px
        total_py = jet_py + lep_py
        total_pz = jet_pz + lep_pz
        invariant_mass = tf.sqrt(
            tf.maximum(total_E**2 - total_px**2 - total_py**2 - total_pz**2, 0.0)
        )

        # Concatenate high-level features
        HLF = tf.concat([delta_R, invariant_mass], axis=-1)
        if jet_lepton_mask is not None:
            jet_lepton_mask = tf.expand_dims(jet_lepton_mask, axis=-1)  # (B, J, L, 1)
            jet_lepton_mask = tf.cast(jet_lepton_mask, tf.bool)
            # Mask invalid pairs
            HLF = tf.where(
                jet_lepton_mask,
                HLF,
                tf.fill(tf.shape(HLF), self.padding_value),
            )        
        return HLF # Shape: (batch_size, n_jets, n_leptons, 2)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "padding_value": self.padding_value,
            }
        )
        return config
