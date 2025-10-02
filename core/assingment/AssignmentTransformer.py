import keras
import tensorflow as tf


from . import MLAssignerBase
from ..components import (
    MultiHeadAttentionBlock,
    SelfAttentionBlock,
    MLP,
    GenerateMask,
    TemporalSoftmax,
    JetLeptonAssignment
)


class CrossAttentionModel(MLAssignerBase):
    def __init__(self, config, name = "CrossAttentionModel"):
        super().__init__(config, name=name)

    def build_model(self, hidden_dim, num_heads, num_encoder_layers, num_decoder_layers, dropout_rate):
        """
        Builds the Assignment Transformer model.
        Args:
            hidden_dim (int): The dimensionality of the hidden layers.
            num_heads (int): The number of attention heads.
            num_layers (int): The number of transformer layers.
            dropout_rate (float): The dropout rate to be applied in the model.
        Returns:
            keras.Model: The constructed Keras model.
        """
        # Input layers
        jet_inputs = keras.Input(shape=(self.max_jets, self.n_jets), name="jet_inputs")
        lep_inputs = keras.Input(shape=(self.max_leptons, self.n_leptons), name="lep_inputs")
        global_inputs = keras.Input(shape=(1,self.n_global), name="global_inputs")

        flatted_global_inputs = keras.layers.Flatten()(global_inputs)

        # Generate masks
        jet_mask = GenerateMask(padding_value=-999, name="jet_mask")(jet_inputs)

        # Add global features to jets and leptons
        global_repeated_jets = keras.layers.RepeatVector(self.max_jets)(flatted_global_inputs)
        jet_features = keras.layers.Concatenate(axis=-1)([jet_inputs, global_repeated_jets])

        global_repeated_leps = keras.layers.RepeatVector(self.max_leptons)(flatted_global_inputs)
        lep_features = keras.layers.Concatenate(axis=-1)([lep_inputs, global_repeated_leps])

        # Input embedding layers
        jet_embedding = MLP(
            hidden_dim,
            num_layers=3,
            dropout_rate=dropout_rate,
            activation="relu",
            name="jet_embedding",
        )(jet_features)

        lep_embedding = MLP(
            hidden_dim,
            num_layers=3,
            dropout_rate=dropout_rate,
            activation="relu",
            name="lep_embedding",
        )(lep_features)

        # Transformer layers
        jet_self_attn = jet_embedding

        for i in range(num_encoder_layers):
            jet_self_attn = SelfAttentionBlock(
                num_heads=num_heads,
                key_dim=hidden_dim,
                dropout_rate=dropout_rate,
                name=f"jets_self_attention_{i}",
            )(jet_self_attn, mask=jet_mask)

        leptons_attent_jets = lep_embedding
        jets_attent_leptons = jet_self_attn
        for i in range(num_decoder_layers):
            leptons_attent_jets = MultiHeadAttentionBlock(
                num_heads=num_heads,
                key_dim=hidden_dim,
                dropout_rate=dropout_rate,
                name=f"leps_cross_attention_{i}",
            )(
                leptons_attent_jets,
                jets_attent_leptons,
                key_mask=jet_mask,
            )
            leptons_attent_jets = MLP(
                hidden_dim,
                num_layers=2,
                dropout_rate=dropout_rate,
                activation="relu",
                name=f"leps_ffn_{i}",
            )(leptons_attent_jets)

        # Output layers
        jet_assignment_probs = JetLeptonAssignment(name="jet_lepton_assignment", dim = hidden_dim)(
            jets_attent_leptons, leptons_attent_jets, jet_mask=jet_mask
        )

        self.model = keras.Model(
            inputs=[jet_inputs, lep_inputs, global_inputs],
            outputs=jet_assignment_probs,
            name="AssignmentTransformer",
        )

class FeatureConcatTransformer(MLAssignerBase):
    def __init__(self, config, name = "FeatureConcatTransformer"):
        super().__init__(config, name=name)

    def build_model(self, hidden_dim, num_heads, num_layers, dropout_rate):
        """
        Builds the Assignment Transformer model.
        Args:
            hidden_dim (int): The dimensionality of the hidden layers.
            num_heads (int): The number of attention heads.
            num_layers (int): The number of transformer layers.
            dropout_rate (float): The dropout rate to be applied in the model.
        Returns:
            keras.Model: The constructed Keras model.
        """
        # Input layers
        jet_inputs = keras.Input(shape=(self.max_jets, self.n_jets), name="jet_inputs")
        lep_inputs = keras.Input(shape=(self.max_leptons, self.n_leptons), name="lep_inputs")
        global_inputs = keras.Input(shape=(1,self.n_global), name="global_inputs")

        flatted_global_inputs = keras.layers.Flatten()(global_inputs)
        flatted_lepton_inputs = keras.layers.Flatten()(lep_inputs)

        # Generate masks
        jet_mask = GenerateMask(padding_value=-999, name="jet_mask")(jet_inputs)


        # Concat global and lepton features to each jet
        global_repeated_jets = keras.layers.RepeatVector(self.max_jets)(flatted_global_inputs)
        lepton_repeated_jets = keras.layers.RepeatVector(self.max_jets)(flatted_lepton_inputs)
        jet_features = keras.layers.Concatenate(axis=-1)([jet_inputs, global_repeated_jets, lepton_repeated_jets])

        # Input embedding layers
        jet_embedding = MLP(
            hidden_dim,
            num_layers=3,
            dropout_rate=dropout_rate,
            activation="relu",
            name="jet_embedding",
        )(jet_features)

        # Transformer layers
        jets_transformed = jet_embedding
        for i in range(num_layers):
            jets_transformed = SelfAttentionBlock(
                num_heads=num_heads,
                key_dim=hidden_dim,
                dropout_rate=dropout_rate,
                name=f"jets_self_attention_{i}",
                pre_ln=True,
            )(jets_transformed, mask=jet_mask)


        # Output layers
        jet_output_embedding = MLP(
            self.max_leptons,
            num_layers=3,
            activation=None,
            name="jet_output_embedding",
        )(jets_transformed)

        jet_assignment_probs = TemporalSoftmax(axis=1, name="jet_assignment_probs")(
            jet_output_embedding, mask=jet_mask
        )
        self.model = keras.Model(
            inputs=[jet_inputs, lep_inputs, global_inputs],
            outputs=jet_assignment_probs,
            name="FeatureConcatTransformer",
        )
