import keras
import tensorflow as tf


from .Assignment import MLAssignerBase, DataConfig
from ..components import (
    MultiHeadAttentionBlock,
    SelfAttentionBlock,
    MLP,
    GenerateMask,
    TemporalSoftmax,
    JetLeptonAssignment,
)


class FeatureConcatRNN(MLAssignerBase):
    def __init__(self, config : DataConfig, name="RNN"):
        super().__init__(config, name)

    def build_model(self, hidden_dim, num_layers, dropout_rate, recurrent_type="lstm"):
        jet_inputs = keras.Input(shape=(self.max_jets, self.n_jets), name="jet_inputs")
        lep_inputs = keras.Input(
            shape=(self.max_leptons, self.n_leptons), name="lep_inputs"
        )
        global_inputs = keras.Input(shape=(1, self.n_global), name="global_inputs")

        flatted_global_inputs = keras.layers.Flatten()(global_inputs)
        flatted_lepton_inputs = keras.layers.Flatten()(lep_inputs)

        # Generate masks
        jet_mask = GenerateMask(padding_value=-999, name="jet_mask")(jet_inputs)

        # Concat lepton and global features to jets
        global_repeated_jets = keras.layers.RepeatVector(self.max_jets)(
            flatted_global_inputs
        )
        lepton_repeated_jets = keras.layers.RepeatVector(self.max_jets)(
            flatted_lepton_inputs
        )
        jet_features = keras.layers.Concatenate(axis=-1)(
            [jet_inputs, global_repeated_jets, lepton_repeated_jets]
        )

        # Input embedding layers
        jet_embedding = MLP(
            hidden_dim,
            num_layers=3,
            dropout_rate=dropout_rate,
            activation="relu",
            name="jet_embedding",
        )(jet_features)

        # Recurrent layers
        if recurrent_type.lower() == "lstm":
            RNNLayer = keras.layers.LSTM
        elif recurrent_type.lower() == "gru":
            RNNLayer = keras.layers.GRU
        else:
            raise ValueError("recurrent_type must be either 'lstm' or 'gru'")

        x = jet_embedding
        for i in range(num_layers):
            x = keras.layers.Bidirectional(RNNLayer(
                hidden_dim,
                return_sequences=True,
                dropout=dropout_rate,
                name=f"bidirectional_rnn_{i}",
            ))(x, mask=jet_mask)
            x = keras.layers.LayerNormalization(name=f"rnn_ln_{i}")(x)

        # Output layers
        jet_assignment_probs = MLP(
            self.max_leptons,
            num_layers=3,
            name="jet_assignment_mlp",
        )(x)

        jet_assignment_probs = TemporalSoftmax(axis=1, name="jet_assignment_probs")(
            jet_assignment_probs, mask=jet_mask
        )
        self.model = keras.Model(
            inputs=[jet_inputs, lep_inputs, global_inputs],
            outputs=jet_assignment_probs,
            name="FeatureConcatRNN",
        )
