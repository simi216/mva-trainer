import keras
import tensorflow as tf
import numpy as np


from core.reconstruction import MLReconstructorBase
from core.components import (
    MultiHeadAttentionBlock,
    SelfAttentionBlock,
    MLP,
    TemporalSoftmax,
    JetLeptonAssignment,
    ComputeHighLevelFeatures,
)


class CrossAttentionModel(MLReconstructorBase):
    def __init__(self, config, name="CrossAttentionModel"):
        super().__init__(config, name=name)
        self.perform_regression = False

    def build_model(
        self,
        hidden_dim,
        num_layers,
        num_heads=8,
        dropout_rate=0.1,
        input_as_four_vector=True,
    ):
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
        normed_jet_inputs, normed_lep_inputs, normed_met_inputs, jet_mask = (
            self._prepare_inputs(input_as_four_vector=input_as_four_vector)
        )

        flatted_met_inputs = keras.layers.Flatten()(normed_met_inputs)

        # Add met features to jets and leptons
        met_repeated_jets = keras.layers.RepeatVector(self.max_jets)(flatted_met_inputs)
        jet_features = keras.layers.Concatenate(axis=-1)(
            [normed_jet_inputs, met_repeated_jets]
        )

        met_repeated_leps = keras.layers.RepeatVector(self.NUM_LEPTONS)(
            flatted_met_inputs
        )
        lep_features = keras.layers.Concatenate(axis=-1)(
            [normed_lep_inputs, met_repeated_leps]
        )

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
        num_encoder_layers = num_layers // 2
        num_decoder_layers = num_layers - num_encoder_layers


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
                dropout_rate=dropout_rate if i < num_decoder_layers - 1 else 0.0,
                name=f"leps_cross_attention_{i}",
            )(
                leptons_attent_jets,
                jets_attent_leptons,
                key_mask=jet_mask,
            )
            jets_attent_leptons = MultiHeadAttentionBlock(
                num_heads=num_heads,
                key_dim=hidden_dim,
                dropout_rate=dropout_rate if i < num_decoder_layers - 1 else 0.0,
                name=f"jets_cross_attention_{i}",
            )(
                jets_attent_leptons,
                leptons_attent_jets,
                key_mask=None,
                query_mask=jet_mask,
            )

        # Output layers
        jet_assignment_probs = JetLeptonAssignment(
            name="assignment", dim=hidden_dim
        )(jets_attent_leptons, leptons_attent_jets, jet_mask=jet_mask)

        self._build_model_base(jet_assignment_probs, name="CrossAttentionModel")


class FeatureConcatTransformer(MLReconstructorBase):
    def __init__(self, config, name="FeatureConcatTransformer", use_nu_flows=True):
        if config.has_regression_targets:
            print("FeatureConcatTransformer is designed for classification tasks; regression targets will be ignored.")
        super().__init__(config, name=name, perform_regression=False, use_nu_flows=use_nu_flows)

    def build_model(
        self,
        hidden_dim,
        num_layers,
        dropout_rate,
        num_heads=8,
        input_as_four_vector=True,
    ):
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
        normed_jet_inputs, normed_lep_inputs, normed_met_inputs, jet_mask = (
            self._prepare_inputs(input_as_four_vector=input_as_four_vector)
        )

        flatted_met_inputs = keras.layers.Flatten()(normed_met_inputs)
        flatted_lepton_inputs = keras.layers.Flatten()(normed_lep_inputs)

        # Concat met and lepton features to each jet
        met_repeated_jets = keras.layers.RepeatVector(self.max_jets)(flatted_met_inputs)
        lepton_repeated_jets = keras.layers.RepeatVector(self.max_jets)(
            flatted_lepton_inputs
        )
        jet_features = keras.layers.Concatenate(axis=-1)(
            [normed_jet_inputs, met_repeated_jets, lepton_repeated_jets]
        )

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
            self.NUM_LEPTONS,
            num_layers=3,
            activation=None,
            name="jet_output_embedding",
        )(jets_transformed)

        jet_assignment_probs = TemporalSoftmax(axis=1, name="assignment")(
            jet_output_embedding, mask=jet_mask
        )
        self._build_model_base(
            jet_assignment_probs, name="FeatureConcatTransformerModel"
        )


class PhysicsInspiredTransformer(MLReconstructorBase):
    def __init__(self, config, name="PhysicsInspiredTransformer"):
        super().__init__(config, name=name)
        self.perform_regression = False
    

    def build_model(
        self,
        hidden_dim,
        num_layers,
        dropout_rate,
        num_heads=8,
        input_as_four_vector=True,
        compute_high_level_features=True,
    ):
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
        normed_jet_inputs, normed_lep_inputs, normed_met_inputs, jet_mask = (
            self._prepare_inputs(input_as_four_vector=input_as_four_vector)
        )

        if compute_high_level_features:
            high_level_feature_layer = ComputeHighLevelFeatures(
                padding_value=self.config.padding_value, name="high_level_features"
            )
            high_level_features = keras.layers.Reshape(target_shape=(self.config.max_jets, -1))(high_level_feature_layer(
                self.transformed_inputs["jets"],
                self.transformed_inputs["leptons"],
                jet_mask=jet_mask,
                lepton_mask=None,
            ))
            normed_jet_inputs = keras.layers.Concatenate(axis=-1)(
                [normed_jet_inputs, high_level_features]
            )

        flatted_met_inputs = keras.layers.Flatten()(normed_met_inputs)
        flatted_lepton_inputs = keras.layers.Flatten()(normed_lep_inputs)

        # Concat met and lepton features to each jet
        met_repeated_jets = keras.layers.RepeatVector(self.max_jets)(flatted_met_inputs)
        lepton_repeated_jets = keras.layers.RepeatVector(self.max_jets)(
            flatted_lepton_inputs
        )
        jet_features = keras.layers.Concatenate(axis=-1)(
            [normed_jet_inputs, met_repeated_jets, lepton_repeated_jets]
        )

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
            self.NUM_LEPTONS,
            num_layers=3,
            activation=None,
            name="jet_output_embedding",
        )(jets_transformed)

        jet_assignment_probs = TemporalSoftmax(axis=1, name="assignment")(
            jet_output_embedding, mask=jet_mask
        )
        self._build_model_base(
            jet_assignment_probs, name="FeatureConcatTransformerModel"
        )

