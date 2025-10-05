from .DataLoader import DataPreprocessor
from .AssignmentBaseModel import AssignmentBaseModel
from .CustomObjects import JetMaskingLayer, TemporalSoftmax
from .components import MLP, SelfAttentionBlock, masking


from keras import layers
import keras


class TransformerJetMatcher(AssignmentBaseModel):
    def __init__(self, preprocessor: DataPreprocessor):
        super().__init__(preprocessor)
        self.data_preprocessor = preprocessor
        print("New TransformerJetMatcher model initialized.")

    def build_model(
        self,
        hidden_size=16,
        lep_dim=4,
        global_dim=2,
        jet_embedding_layers=2,
        lepton_embedding_layers=2,
        global_embedding_layers=2,
        attention_blocks=3,
        ff_layers=2,
        n_heads=4,
        dropout_rate=0.1,
        regularization_lambda=0.001,
    ):
        """
        Build the Transformer model for jet matching.
        """

        # Define input shapes
        jet_input = keras.Input(shape=(self.max_jets, self.n_jets), name="jet")
        lepton_input = keras.Input(
            shape=(self.max_leptons, self.n_leptons), name="lepton"
        )
        global_input = keras.Input(shape=(1, self.n_global), name="global")

        jet_mask = JetMaskingLayer(padding_value=self.padding_value, name="jet_mask")(
            jet_input
        )
        jet_mask_reshaped = layers.Reshape(
            (self.max_jets, 1), name="jet_mask_reshaped"
        )(jet_mask)

        # Jet embedding
        jet_embedding = jet_input
        for i in range(jet_embedding_layers):
            jet_embedding = layers.Dense(
                hidden_size,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(regularization_lambda),
                name=f"jet_embedding_{i+1}",
            )(jet_embedding)
            jet_embedding = layers.Dropout(
                rate=dropout_rate, name=f"jet_embedding_dropout_{i+1}"
            )(jet_embedding)

        # Lepton embedding
        lepton_embedding = lepton_input
        for i in range(lepton_embedding_layers):
            lepton_embedding = layers.Dense(
                lep_dim,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(regularization_lambda),
                name=f"lepton_embedding_{i+1}",
            )(lepton_embedding)
            lepton_embedding = layers.Dropout(
                rate=dropout_rate, name=f"lepton_embedding_dropout_{i+1}"
            )(lepton_embedding)

        # Global embedding
        global_embedding = global_input
        for i in range(global_embedding_layers):
            global_embedding = layers.Dense(
                global_dim,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(regularization_lambda),
                name=f"global_embedding_{i+1}",
            )(global_embedding)
            global_embedding = layers.Dropout(
                rate=dropout_rate, name=f"global_embedding_dropout_{i+1}"
            )(global_embedding)

        # Concatenate jet, lepton, and global embeddings
        global_embedding_repeated = layers.Reshape((global_dim,))(global_embedding)
        global_embedding_repeated = layers.RepeatVector(self.max_jets)(
            global_embedding_repeated
        )

        lepton_embedding_repeated = layers.Reshape((lep_dim * self.max_leptons,))(
            lepton_embedding
        )
        lepton_embedding_repeated = layers.RepeatVector(self.max_jets)(
            lepton_embedding_repeated
        )
        jet_lepton_global_concat = layers.Concatenate(
            axis=-1, name="jet_lepton_global_concat"
        )([jet_embedding, lepton_embedding_repeated, global_embedding_repeated])

        # Jet-lepton-global embedding
        jet_lepton_global_embedding = jet_lepton_global_concat
        for i in range(jet_embedding_layers):
            jet_lepton_global_embedding = layers.Dense(
                hidden_size,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(regularization_lambda),
                name=f"jet_lepton_global_embedding_{i+1}",
            )(jet_lepton_global_embedding)
            jet_lepton_global_embedding = layers.Dropout(
                rate=dropout_rate, name=f"jet_lepton_global_embedding_dropout_{i+1}"
            )(jet_lepton_global_embedding)

        # Transformer blocks
        attention_output = jet_lepton_global_embedding
        for i in range(attention_blocks):
            attention_output = layers.MultiHeadAttention(
                num_heads=n_heads,
                key_dim=hidden_size,
                kernel_regularizer=keras.regularizers.l2(regularization_lambda),
                name=f"attention_block_{i+1}",
            )(
                attention_output,
                attention_output,
                attention_output,
                attention_mask=jet_mask,
            )
            attention_output = layers.Dropout(
                rate=dropout_rate, name=f"attention_block_{i+1}_dropout"
            )(attention_output)
            attention_output = layers.LayerNormalization(
                name=f"attention_block_{i+1}_norm"
            )(attention_output)
            attention_output = layers.Add(name=f"attention_block_{i+1}_add")(
                [attention_output, jet_embedding]
            )
            # Feed-forward network
            for j in range(ff_layers):
                attention_output = layers.Dense(
                    hidden_size, activation="relu", name=f"ff_layer_{i+1}_{j+1}"
                )(attention_output)
                attention_output = layers.Dropout(
                    rate=dropout_rate, name=f"ff_layer_{i+1}_{j+1}_dropout"
                )(attention_output)
            attention_output = layers.LayerNormalization(
                name=f"ff_layer_{i+1}_norm_ff"
            )(attention_output)

        # Lepton 1 head
        lep_1_head = attention_output
        for i in range(jet_embedding_layers):
            lep_1_head = layers.Dense(
                hidden_size,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(regularization_lambda),
                name=f"output_layer_lep_1_{i+1}",
            )(lep_1_head)
        lep_1_head = layers.Dense(1, activation="linear", name="output")(lep_1_head)
        lep_1_output = TemporalSoftmax(name="temporal_softmax")(
            lep_1_head, mask=jet_mask_reshaped
        )

        # Lepton 2 head
        lep_2_head = attention_output
        for i in range(jet_embedding_layers):
            lep_2_head = layers.Dense(
                hidden_size,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(regularization_lambda),
                name=f"output_layer_lep_2_{i+1}",
            )(lep_2_head)
        lep_2_head = layers.Dense(1, activation="linear", name="output_lep_2")(
            lep_2_head
        )
        lep_2_output = TemporalSoftmax(name="temporal_softmax_lep_2")(
            lep_2_head, mask=jet_mask_reshaped
        )

        # Combine outputs
        output = layers.Concatenate(axis=-1, name="final_output")(
            [lep_1_output, lep_2_output]
        )

        # Create the model
        self.model = keras.Model(
            inputs=[jet_input, lepton_input, global_input],
            outputs=output,
            name="TransformerJetMatcher",
        )


class FeatureConcatTransformer(AssignmentBaseModel):
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
        jet_mask = masking.GenerateMask(padding_value=-999, name="jet_mask")(jet_inputs)


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

        jet_assignment_probs = masking.TemporalSoftmax(axis=1, name="jet_assignment_probs")(
            jet_output_embedding, mask=jet_mask
        )
        self.model = keras.Model(
            inputs=[jet_inputs, lep_inputs, global_inputs],
            outputs=jet_assignment_probs,
            name="FeatureConcatTransformer",
        )