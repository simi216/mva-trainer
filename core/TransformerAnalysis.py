from .DataLoader import DataPreprocessor
from .BaseModel import BaseModel
from .CustomObjects import JetMaskingLayer, TemporalSoftmax

from keras import layers
import keras


class TransformerJetMatcher(BaseModel):
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
            lep_1_head = layers.Dropout(
                rate=dropout_rate, name=f"output_layer_lep_1_{i+1}_dropout"
            )(lep_1_head)

        lep_1_head = layers.Dense(1, activation="sigmoid", name="output")(lep_1_head)
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
            lep_2_head = layers.Dropout(
                rate=dropout_rate, name=f"output_layer_lep_2_{i+1}_dropout"
            )(lep_2_head)
        lep_2_head = layers.Dense(1, activation="sigmoid", name="output_lep_2")(
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


class SimpleTransformer(BaseModel):
    def __init__(self, preprocessor: DataPreprocessor):
        super().__init__(preprocessor)
        self.data_preprocessor = preprocessor
        print("New SimpleTransformer model initialized.")

    def build_model(
        self,
        hidden_size=32,
        attention_stacks=3,
        attention_heads=8,
        ff_layers=2,
        dropout_rate=0.1,
        regularization_lambda=0.001,
    ):
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
        jet_mask_flatted = layers.Flatten(name="jet_mask_flatted")(jet_mask_reshaped)

        # Concatenate inputs
        lepton_input_repeated = layers.RepeatVector(self.max_jets)(
            layers.Reshape((self.max_leptons * self.n_leptons,))(lepton_input)
        )
        global_input_repeated = layers.RepeatVector(self.max_jets)(
            layers.Reshape((self.n_global,))(global_input)
        )
        concatenated_inputs = layers.Concatenate(axis=-1, name="concatenated_inputs")(
            [jet_input, lepton_input_repeated, global_input_repeated]
        )

        # Embedding layers
        for i in range(3):
            concatenated_inputs = layers.Dense(
                hidden_size,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(regularization_lambda),
                name=f"embedding_layer_{i+1}",
            )(concatenated_inputs)
            concatenated_inputs = layers.Dropout(
                rate=dropout_rate, name=f"embedding_layer_{i+1}_dropout"
            )(concatenated_inputs)

        # Transformer blocks
        transformer_output = concatenated_inputs
        for i in range(attention_stacks):
            transformer_residual = transformer_output
            transformer_output = layers.MultiHeadAttention(
                num_heads=attention_heads,
                key_dim=hidden_size,
                name=f"attention_block_{i+1}",
                kernel_regularizer=keras.regularizers.l2(regularization_lambda),
            )(
                transformer_output,
                transformer_output,
                attention_mask=jet_mask,
            )
            transformer_output = layers.Dropout(
                rate=dropout_rate, name=f"attention_block_{i+1}_dropout"
            )(transformer_output)
            transformer_output = layers.LayerNormalization(
                name=f"attention_block_{i+1}_norm"
            )(transformer_output)
            transformer_output = layers.Add(name=f"attention_block_{i+1}_add")(
                [transformer_output, transformer_residual]
            )
            # Feed-forward network
            for j in range(ff_layers):
                transformer_output = layers.Dense(
                    hidden_size,
                    activation="relu",
                    name=f"ff_layer_{i+1}_{j+1}",
                    kernel_regularizer=keras.regularizers.l2(regularization_lambda),
                )(transformer_output)
                transformer_output = layers.Dropout(
                    rate=dropout_rate, name=f"ff_layer_{i+1}_{j+1}_dropout"
                )(transformer_output)
            transformer_output = layers.LayerNormalization(
                name=f"ff_layer_{i+1}_norm_ff"
            )(transformer_output)
        # Output layers
        output = transformer_output
        for i in range(ff_layers):
            output = layers.Dense(
                hidden_size,
                activation="relu",
                name=f"output_layer_{i+1}",
                kernel_regularizer=keras.regularizers.l2(regularization_lambda),
            )(output)
            output = layers.Dropout(
                rate=dropout_rate, name=f"output_layer_{i+1}_dropout"
            )(output)
        output = layers.Dense(2, activation="sigmoid", name="output")(output)
        final_output = TemporalSoftmax(name="temporal_softmax")(
            output, mask=jet_mask_reshaped
        )
        # Create the model
        self.model = keras.Model(
            inputs=[jet_input, lepton_input, global_input],
            outputs=final_output,
            name="SimpleTransformer",
        )
