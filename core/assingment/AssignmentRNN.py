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

    def build_model(self, hidden_dim, num_layers, dropout_rate, recurrent_type="lstm",input_as_four_vector=True):

        # Input layers
        jet_inputs, lep_inputs, met_inputs, jet_mask = self._prepare_inputs(input_as_four_vector=input_as_four_vector)

        flatted_met_inputs = keras.layers.Flatten()(met_inputs)
        flatted_lepton_inputs = keras.layers.Flatten()(lep_inputs)

        # Concat lepton and met features to jets
        met_repeated_jets = keras.layers.RepeatVector(self.max_jets)(
            flatted_met_inputs
        )
        lepton_repeated_jets = keras.layers.RepeatVector(self.max_jets)(
            flatted_lepton_inputs
        )
        jet_features = keras.layers.Concatenate(axis=-1)(
            [jet_inputs, met_repeated_jets, lepton_repeated_jets]
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
        # Build model
        self._build_model_base(jet_assignment_probs, name="FeatureConcatRNN")