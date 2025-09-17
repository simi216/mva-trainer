import keras
import tensorflow as tf


from .AssignmentBaseModel import AssignmentBaseModel
from ..components import (
    MultiHeadAttentionBlock,
    SelfAttentionBlock,
    MLP,
    GenerateMask,
    TemporalSoftmax,
    JetLeptonAssignment
)


class CrossAttentionModel(AssignmentBaseModel):
    def __init__(self, data_preprocessor):
        super().__init__(data_preprocessor)

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
        jets_attent_leptons = jet_embedding
        leptons_attent_jets = lep_embedding
        for i in range(num_layers):
            jets_attent_leptons = MultiHeadAttentionBlock(
                num_heads=num_heads,
                key_dim=hidden_dim,
                dropout_rate=dropout_rate,
                name=f"jets_attent_leptons_{i}",
            )(jets_attent_leptons, leptons_attent_jets, query_mask=jet_mask)

            leptons_attent_jets = MultiHeadAttentionBlock(
                num_heads=num_heads,
                key_dim=hidden_dim,
                dropout_rate=dropout_rate,
                name=f"leptons_attent_jets_{i}",
            )(leptons_attent_jets, jets_attent_leptons, value_mask=jet_mask)

            jets_attent_leptons = SelfAttentionBlock(
                num_heads=num_heads,
                key_dim=hidden_dim,
                dropout_rate=dropout_rate,
                name=f"jets_self_attention_{i}",
            )(jets_attent_leptons, mask=jet_mask)

        # Output layers
        jet_assignment_probs = JetLeptonAssignment(name="jet_lepton_assignment", dim = hidden_dim)(
            jets_attent_leptons, leptons_attent_jets, jet_mask=jet_mask
        )

        self.model = keras.Model(
            inputs=[jet_inputs, lep_inputs, global_inputs],
            outputs=jet_assignment_probs,
            name="AssignmentTransformer",
        )

class FeatureConcatTransformer(AssignmentBaseModel):
    def __init__(self, data_preprocessor):
        super().__init__(data_preprocessor)

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
            )(jets_transformed, mask=jet_mask)
        
        # Output layers
        jet_outputs = MLP(
            hidden_dim,
            num_layers=3,
            dropout_rate=dropout_rate,
            activation="relu",
            name="jet_outputs_mlp",
        )(jets_transformed)

        jet_output_embedding = MLP(
            self.max_leptons,
            num_layers=2,
            dropout_rate=0.0,
            activation=None,
            name="jet_output_embedding",
        )(jet_outputs)

        jet_assignment_probs = TemporalSoftmax(axis=1, name="jet_assignment_probs")(
            jet_output_embedding, mask=jet_mask
        )
        self.model = keras.Model(
            inputs=[jet_inputs, lep_inputs, global_inputs],
            outputs=jet_assignment_probs,
            name="FeatureConcatTransformer",
        )
