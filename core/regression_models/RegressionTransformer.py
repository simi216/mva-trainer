import keras
import tensorflow as tf
import numpy as np


from core.reconstruction import FFMLRecoBase
from core.components import (
    SelfAttentionBlock,
    MLP,
    TemporalSoftmax,
    PoolingAttentionBlock,
)


class FeatureConcatTransformer(FFMLRecoBase):
    def __init__(self, config, name="FeatureConcatTransformer"):
        super().__init__(config, name=name, perform_regression=True, use_nu_flows=False)

    def build_model(
        self,
        hidden_dim,
        central_transformer_stack_size,
        regression_transformer_stack_size,
        dropout_rate,
        num_heads=8,
        compute_HLF=False,
        log_variables=False,
        use_global_event_features=False,
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
        normed_inputs, masks = self._prepare_inputs(
            compute_HLF=compute_HLF,
            log_variables=log_variables,
            use_global_event_features=use_global_event_features,
        )
        normed_jet_inputs = normed_inputs["jet_inputs"]
        normed_lep_inputs = normed_inputs["lepton_inputs"]
        normed_met_inputs = normed_inputs["met_inputs"]
        jet_mask = masks["jet_mask"]

        if compute_HLF:
            normed_HLF_inputs = normed_inputs["hlf_inputs"]
            flat_normed_HLF_inputs = keras.layers.Reshape((self.max_jets, -1))(
                normed_HLF_inputs
            )
            normed_jet_inputs = keras.layers.Concatenate(axis=-1)(
                [normed_jet_inputs, flat_normed_HLF_inputs]
            )

        if self.config.has_global_event_features:
            normed_global_event_inputs = normed_inputs["global_event_inputs"]
            flatted_global_event_inputs = keras.layers.Flatten()(
                normed_global_event_inputs
            )
            # Add global event features to jets
            global_event_repeated_jets = keras.layers.RepeatVector(self.max_jets)(
                flatted_global_event_inputs
            )
            normed_jet_inputs = keras.layers.Concatenate(axis=-1)(
                [normed_jet_inputs, global_event_repeated_jets]
            )

        flatted_met_inputs = keras.layers.Flatten()(normed_met_inputs)
        flatted_lepton_inputs = keras.layers.Flatten()(normed_lep_inputs)

        # Concat met and lepton features to each jet
        met_repeated_jets = keras.layers.RepeatVector(self.max_jets)(flatted_met_inputs)
        lepton_repeated_jets = keras.layers.RepeatVector(self.max_jets)(
            flatted_lepton_inputs
        )
        jet_concat_features = keras.layers.Concatenate(axis=-1)(
            [normed_jet_inputs, met_repeated_jets, lepton_repeated_jets]
        )

        # Input embedding layers
        jet_embedding = MLP(
            hidden_dim,
            num_layers=3,
            dropout_rate=dropout_rate,
            activation="relu",
            name="jet_embedding",
        )(jet_concat_features)

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

        projected_jet_transformed = MLP(
            hidden_dim
            - normed_met_inputs.shape[-1]
            - self.NUM_LEPTONS * normed_lep_inputs.shape[-1]
            - normed_jet_inputs.shape[-1]
            - self.NUM_LEPTONS,
            num_layers=3,
            dropout_rate=dropout_rate,
            activation="relu",
            name="projected_jet_transformed",
        )(jets_transformed)

        regression_input = keras.layers.Concatenate(axis=-1)(
            [
                projected_jet_transformed,
                jet_assignment_probs,
                keras.layers.RepeatVector(self.max_jets)(flatted_lepton_inputs),
                keras.layers.RepeatVector(self.max_jets)(flatted_met_inputs),
                normed_jet_inputs,
            ]
        )

        regression_input = MLP(
            hidden_dim,
            num_layers=3,
            dropout_rate=dropout_rate,
            activation="relu",
            name="regression_input_mlp",
        )(regression_input)

        # Neutrino momentum head
        neutrino_momentum_head = PoolingAttentionBlock(
            key_dim=hidden_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            num_seeds=self.NUM_LEPTONS,
            pre_ln=True,
            name="neutrino_momentum_head",
        )(
            regression_input,
            mask=jet_mask,
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
            self.config.get_n_neutrino_truth(),
            num_layers=4,
            activation=None,
            dropout_rate=dropout_rate,
            name="normalized_regression",
        )(neutrino_self_attention)

        # Build the final model
        self._build_model_base(
            jet_assignment_probs,
            neutrino_momentum_outputs,
            name="FeatureConcatTransformerModel",
        )
