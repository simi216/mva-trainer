import keras
import tensorflow as tf
import numpy as np


from core.reconstruction import MLReconstructorBase
from core.components import (
    SelfAttentionBlock,
    MLP,
    TemporalSoftmax,
    PoolingAttentionBlock,
)


class FeatureConcatTransformer(MLReconstructorBase):
    def __init__(self, config, name="FeatureConcatTransformer"):
        super().__init__(config, name=name, perform_regression=True, use_nu_flows=False)

    def build_model(
        self,
        hidden_dim,
        central_transformer_stack_size,
        regression_transformer_stack_size,
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
        for i in range(central_transformer_stack_size):
            jets_transformed = SelfAttentionBlock(
                num_heads=num_heads,
                key_dim=hidden_dim,
                dropout_rate=dropout_rate,
                name=f"jets_self_attention_{i}",
                pre_ln=True,
            )(jets_transformed, mask=jet_mask)

        # Assignment ouput
        jet_output_embedding = MLP(
            self.NUM_LEPTONS,
            num_layers=3,
            activation=None,
            name="jet_output_embedding",
        )(jets_transformed)

        jet_assignment_probs = TemporalSoftmax(axis=1, name="assignment")(
            jet_output_embedding, mask=jet_mask
        )

        # Neutrino momentum head
        neutrino_momentum_head = PoolingAttentionBlock(
            key_dim=hidden_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            num_seeds=self.NUM_LEPTONS,
            pre_ln=True,
            name="neutrino_momentum_head",
        )(
            jets_transformed,
            mask=jet_mask,
        )
        # MET Residual Connection
        neutrino_residual = keras.layers.RepeatVector(
            self.config.NUM_LEPTONS, name="neutrino_residual"
        )(
            keras.layers.Dense(
                hidden_dim, activation="relu", name="neutrino_residual_embedding"
            )(flatted_met_inputs)
        )
        neutrino_momentum_head = keras.layers.Add(name="neutrino_residual_add")(
            [neutrino_momentum_head, neutrino_residual]
        )

        neutrino_self_attention = neutrino_momentum_head
        for i in range(regression_transformer_stack_size):
            neutrino_self_attention = SelfAttentionBlock(
                num_heads=num_heads,
                key_dim=hidden_dim,
                dropout_rate=dropout_rate,
                name=f"neutrino_self_attention_{i}",
                pre_ln=True,
            )(neutrino_self_attention)

        neutrino_momentum_outputs = MLP(
            self.config.get_n_regression_targets(),
            num_layers=4,
            activation=None,
            dropout_rate=dropout_rate,
            name="regression",
        )(neutrino_self_attention)

        # Build the final model
        self._build_model_base(
            jet_assignment_probs,
            neutrino_momentum_outputs,
            name="FeatureConcatTransformerModel",
        )


class SimpleNeutrinoRegessor(MLReconstructorBase):
    def __init__(self, config, name="SimpleNeutrinoRegessor"):
        super().__init__(config, name=name)
        self.perform_regression = True

    def build_model(
        self,
        hidden_dim,
        central_transformer_stack_size,
        neutrino_regression_layers,
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
        for i in range(central_transformer_stack_size):
            jets_transformed = SelfAttentionBlock(
                num_heads=num_heads,
                key_dim=hidden_dim,
                dropout_rate=dropout_rate,
                name=f"jets_self_attention_{i}",
                pre_ln=True,
            )(jets_transformed, mask=jet_mask)

        # Assignment ouput
        jet_output_embedding = MLP(
            self.NUM_LEPTONS,
            num_layers=3,
            activation=None,
            name="jet_output_embedding",
        )(jets_transformed)

        jet_assignment_probs = TemporalSoftmax(axis=1, name="assignment")(
            jet_output_embedding, mask=jet_mask
        )

        # Neutrino momentum head
        neutrino_momentum_head = PoolingAttentionBlock(
            key_dim=hidden_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            num_seeds=self.NUM_LEPTONS,
            pre_ln=True,
            name="neutrino_momentum_head",
        )(
            jets_transformed,
            mask=jet_mask,
        )
        # MET Residual Connection
        neutrino_residual = keras.layers.RepeatVector(
            self.config.NUM_LEPTONS, name="neutrino_residual"
        )(
            keras.layers.Dense(
                hidden_dim, activation="relu", name="neutrino_residual_embedding"
            )(flatted_met_inputs)
        )
        neutrino_momentum_head = keras.layers.Add(name="neutrino_residual_add")(
            [neutrino_momentum_head, neutrino_residual]
        )

        neutrino_momentum_flat = keras.layers.Flatten()(neutrino_momentum_head)

        neutrino_momentum_outputs_flat = MLP(
            2 * hidden_dim,
            num_layers=neutrino_regression_layers,
            activation=None,
            dropout_rate=dropout_rate,
            name="regression_intermediate",
        )(neutrino_momentum_flat)

        neutrino_momentum_outputs_flat = MLP(
            self.config.NUM_LEPTONS * self.config.get_n_regression_targets(),
            num_layers=neutrino_regression_layers,
            activation=None,
            dropout_rate=dropout_rate,
            name="regression_final",
        )(neutrino_momentum_outputs_flat)

        neutrino_momentum_outputs = keras.layers.Reshape(
            (self.config.NUM_LEPTONS, self.config.get_n_regression_targets()),
            name="regression_unscaled_reshaped",
        )(neutrino_momentum_outputs_flat)

        scaled_neutrino_momentum_outputs = keras.layers.Rescaling(
            1e6, name="regression"
        )(neutrino_momentum_outputs)


        # Build the final model
        self._build_model_base(
            jet_assignment_probs,
            scaled_neutrino_momentum_outputs,
            name="FeatureConcatTransformerModel",
        )
