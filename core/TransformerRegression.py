from .DataLoader import DataPreprocessor
from .RegressionBaseModel import RegressionBaseModel
from .CustomObjects import JetMaskingLayer, TemporalSoftmax, SplitLayer, DataNormalizationLayer, AppendTrueMask, PrependLearnedVector
import numpy as np
import pandas as pd

import tensorflow as tf
import keras
from keras.src.api_export import keras_export
from keras import layers
import sklearn as sk
import onnx


class RegressionTransformer(RegressionBaseModel):
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
        jet_mask_flatted = layers.Flatten(name="jet_mask_flatted")(jet_mask_reshaped)

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

        # Normalize inputs
        normed_jet_input = DataNormalizationLayer(name="jet_input_norm", data = self.X_train["jet"], axis = 2)(jet_input)
        normed_lepton_input = DataNormalizationLayer(name="lepton_input_norm", data = self.X_train["lepton"], axis = 2)(lepton_input)
        normed_global_input = DataNormalizationLayer(name="global_input_norm", data = self.X_train["global"], axis = 2)(global_input)


        # Concatenate jet, lepton, and global embeddings
        global_embedding_repeated = layers.Reshape((global_dim,))(normed_global_input)
        global_embedding_repeated = layers.RepeatVector(self.max_jets)(
            global_embedding_repeated
        )

        lepton_embedding_repeated = layers.Reshape((lep_dim * self.max_leptons,))(
            normed_lepton_input
        )
        lepton_embedding_repeated = layers.RepeatVector(self.max_jets)(
            lepton_embedding_repeated
        )
        jet_lepton_global_concat = layers.Concatenate(
            axis=-1, name="jet_lepton_global_concat"
        )([normed_jet_input, lepton_embedding_repeated, global_embedding_repeated])

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
        assignment_output = layers.Concatenate(axis=-1, name="assignment_output")(
            [lep_1_output, lep_2_output]
        )

        # Regression output
        global_regression_embedding = global_input
        for i in range(global_embedding_layers):
            global_regression_embedding = layers.Dense(
                global_dim, activation="relu", name=f"global_regression_embedding_{i+1}"
            )(global_regression_embedding)

        lepton_regression_embedding = lepton_input
        for i in range(lepton_embedding_layers):
            lepton_regression_embedding = layers.Dense(
                lep_dim, activation="relu", name=f"lepton_regression_embedding_{i+1}"
            )(lepton_regression_embedding)

        jet_regression_embedding = jet_input
        for i in range(jet_embedding_layers):
            jet_regression_embedding = layers.Dense(
                hidden_size, activation="relu", name=f"jet_regression_embedding_{i+1}"
            )(jet_regression_embedding)

        global_regression_embedding_flatted = layers.Flatten()(
            global_regression_embedding
        )
        global_regression_embedding_repeated = layers.RepeatVector(self.max_jets)(
            global_regression_embedding_flatted
        )
        lepton_regression_embedding_flattened = layers.Flatten()(
            lepton_regression_embedding
        )
        lepton_regression_embedding_repeated = layers.RepeatVector(self.max_jets)(
            lepton_regression_embedding_flattened
        )

        regression_concat = layers.Concatenate(axis=-1, name="regression_concat")(
            [
                jet_regression_embedding,
                lepton_regression_embedding_repeated,
                global_regression_embedding_repeated,
                assignment_output,
            ]
        )

        regression_output = regression_concat
        for i in range(jet_embedding_layers):
            regression_output = layers.Dense(
                hidden_size, activation="relu", name=f"regression_output_layer_{i+1}"
            )(regression_output)

        regression_attention_output = regression_output
        for i in range(attention_blocks):
            regression_attention_output = layers.MultiHeadAttention(
                num_heads=n_heads,
                key_dim=hidden_size,
                name=f"regression_attention_block_{i+1}",
            )(
                regression_attention_output,
                regression_attention_output,
                regression_attention_output,
                attention_mask=jet_mask,
            )
            regression_attention_output = layers.LayerNormalization(
                name=f"regression_attention_block_{i+1}_norm"
            )(regression_attention_output)
            regression_attention_output = layers.Add(
                name=f"regression_attention_block_{i+1}_add"
            )([regression_attention_output, regression_output])
            # Feed-forward network
            for j in range(ff_layers):
                regression_attention_output = layers.Dense(
                    hidden_size,
                    activation="relu",
                    name=f"regression_ff_layer_{i+1}_{j+1}",
                )(regression_attention_output)
            regression_attention_output = layers.LayerNormalization(
                name=f"regression_ff_layer_{i+1}_norm_ff"
            )(regression_attention_output)

        regression_output = layers.GlobalAveragePooling1D(
            name="global_average_pooling"
        )(regression_attention_output, mask=jet_mask_flatted)
        for i in range(jet_embedding_layers):
            regression_output = layers.Dense(
                hidden_size, activation="relu", name=f"regression_output_dense_{i+1}"
            )(regression_output)
        regression_output = layers.Dense(
            1, activation="linear", name="regression_output"
        )(regression_output)

        self.model = keras.Model(
            inputs=[jet_input, lepton_input, global_input],
            outputs=[assignment_output, regression_output],
            name="RegressionTransformerModel",
        )

class SequentialRegression(RegressionBaseModel):
    def build_model(
        self,
        hidden_size=32,
        attention_stacks=3,
        attention_heads=8,
        ff_layers=2,
        dropout_rate=0.1,
        regression_dim = 32,
        regression_attention_stacks = 2,
        regression_ff_layers = 2,
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
        assignment_output = TemporalSoftmax(name="assignment_output")(
            output, mask=jet_mask_reshaped
        )

        # Regression output
        regression_concat = layers.Concatenate(name = "regression_concat", axis = -1)(
            [jet_input, lepton_input_repeated, global_input_repeated, assignment_output]
        )
        regression_output = regression_concat
        i = 0
        dim = self.n_jets + self.n_leptons * self.max_leptons + self.n_global + 2
        while(dim < regression_dim):
            regression_output = layers.Dense(
                dim,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(regularization_lambda),
                name=f"regression_dense_{i+1}",
            )(regression_output)
            regression_output = layers.Dropout(
                rate=dropout_rate, name=f"regression_dropout_{i+1}"
            )(regression_output)
            dim *= 2
            i += 1
        i += 1
        regression_output = layers.Dense(
            regression_dim,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(regularization_lambda),
            name=f"regression_dense_{i+1}",
        )(regression_output)
        regression_output = layers.Dropout(
            rate=dropout_rate, name=f"regression_dropout_{i+1}"
        )(regression_output)

        for i in range(regression_attention_stacks):
            regression_residual = regression_output
            regression_output = layers.MultiHeadAttention(
                num_heads=attention_heads,
                key_dim=regression_dim,
                name=f"regression_attention_block_{i+1}",
                kernel_regularizer=keras.regularizers.l2(regularization_lambda),
            )(
                regression_output,
                regression_output,
                attention_mask=jet_mask,
            )
            regression_output = layers.Dropout(
                rate=dropout_rate, name=f"regression_attention_block_{i+1}_dropout"
            )(regression_output)
            regression_output = layers.LayerNormalization(
                name=f"regression_attention_block_{i+1}_norm"
            )(regression_output)
            regression_output = layers.Add(name=f"regression_attention_block_{i+1}_add")(
                [regression_output, regression_residual]
            )
            # Feed-forward network
            for j in range(regression_ff_layers):
                regression_output = layers.Dense(
                    regression_dim,
                    activation="relu",
                    name=f"regression_ff_layer_{i+1}_{j+1}",
                    kernel_regularizer=keras.regularizers.l2(regularization_lambda),
                )(regression_output)
                regression_output = layers.Dropout(
                    rate=dropout_rate, name=f"regression_ff_layer_{i+1}_{j+1}_dropout"
                )(regression_output)
            regression_output = layers.LayerNormalization(
                name=f"regression_ff_layer_{i+1}_norm_ff"
            )(regression_output)

        regression_output = layers.GlobalAveragePooling1D(
            name="regression_global_average_pooling"
        )(regression_output, mask=jet_mask_flatted)
        i = 0
        dim = regression_dim
        while dim > self.n_regression_targets:
            regression_output = layers.Dense(
                dim,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(regularization_lambda),
                name=f"regression_dense_final_{i+1}",
            )(regression_output)
            regression_output = layers.Dropout(
                rate=dropout_rate, name=f"regression_dropout_final_{i+1}"
            )(regression_output)
            dim //= 2
            i += 1
        
        final_regression_output = layers.Dense(
            self.n_regression_targets,
            activation="linear",
            kernel_regularizer=keras.regularizers.l2(regularization_lambda),
            name="regression_output",
        )(regression_output)

        # Create the model
        self.model = keras.Model(
            inputs=[jet_input, lepton_input, global_input],
            outputs={"assignment_output": assignment_output, "regression_output": final_regression_output},
            name="SequentialRegressionModel",
        )


class SimpleRegression(RegressionBaseModel):
    def build_model(
        self,
        hidden_size=16,
        number_attention_heads=8,
        transformer_layers=3,
        ff_layers=2,
        dropout_rate=0.1,
        regularization_lambda=0.001,
        regression_activation="linear",
    ):
        """
        Build a simple regression model with dense layers and dropout.
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

        jet_mask_flatted = layers.Flatten(name="jet_mask_flatted")(jet_mask_reshaped)

        # Normalize inputs
        jet_data_mask = np.all(self.X_train["jet"] != self.padding_value, axis=-1)
        masked_jet_data = self.X_train["jet"][jet_data_mask,:]
        masked_jet_data = np.broadcast_to(masked_jet_data.reshape(masked_jet_data.shape[0], 1, self.n_jets), (masked_jet_data.shape[0], self.max_jets, self.n_jets))

        normed_jet_input = DataNormalizationLayer(name="jet_input_norm", data = masked_jet_data, axis = 2)(jet_input)
        normed_lepton_input = DataNormalizationLayer(name="lepton_input_norm", data = self.X_train["lepton"], axis = 2)(lepton_input)
        normed_global_input = DataNormalizationLayer(name="global_input_norm", data = self.X_train["global"], axis = 2)(global_input)


        # Concatenate jet, lepton, and global embeddings
        global_embedding_repeated = layers.Flatten()(normed_global_input)
        global_embedding_repeated = layers.RepeatVector(self.max_jets)(
            global_embedding_repeated
        )

        lepton_embedding_repeated = layers.Flatten()(
            normed_lepton_input
        )
        lepton_embedding_repeated = layers.RepeatVector(self.max_jets)(
            lepton_embedding_repeated
        )

        concatenated_inputs = layers.Concatenate(
            axis=-1, name="jet_lepton_global_concat"
        )([normed_jet_input, lepton_embedding_repeated, global_embedding_repeated])

        # Dense layers for input processing
        dense_output = concatenated_inputs
        dim = self.n_jets + self.n_leptons * self.max_leptons + self.n_global
        i = 0
        while dim < hidden_size:
            dense_output = layers.Dense(
                dim,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(regularization_lambda),
                name=f"dense_layer_{i+1}",
            )(dense_output)
            dense_output = layers.Dropout(
                rate=dropout_rate, name=f"dense_dropout_{i+1}"
            )(dense_output)
            dim *= 2
            i += 1

        dense_output = layers.Dense(
            hidden_size,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(regularization_lambda),
            name=f"dense_layer_{i+1}",
        )(dense_output)
        dense_output = layers.Dropout(rate=dropout_rate, name=f"dense_dropout_{i+1}")(
            dense_output
        )
        dense_output = layers.LayerNormalization(
            epsilon=1e-6, name="dense_layer_norm"
        )(dense_output)

        # Transformer blocks
        transformer_output = dense_output
        for i in range(transformer_layers):
            # Multi-head self-attention
            attention_output = layers.MultiHeadAttention(
                num_heads=number_attention_heads,
                key_dim=hidden_size,
                dropout=dropout_rate,
                name=f"attention_block_{i+1}",
            )(transformer_output, transformer_output, attention_mask=jet_mask)
            attention_output = layers.Dropout(
                rate=dropout_rate, name=f"attention_dropout_{i+1}"
            )(attention_output)
            attention_output = layers.LayerNormalization(
                epsilon=1e-6, name=f"attention_layernorm_{i+1}"
            )(attention_output + transformer_output)

            # Residual connection
            attention_output = layers.Dropout(
                rate=dropout_rate, name=f"attention_residual_dropout_{i+1}"
            )(attention_output)
            attention_output = layers.LayerNormalization(
                epsilon=1e-6, name=f"attention_residual_layernorm_{i+1}"
            )(attention_output + transformer_output)

            # Feed-forward network
            ff_output = attention_output
            for j in range(ff_layers):
                ff_output = layers.Dense(
                    hidden_size * 2,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l2(regularization_lambda),
                    name=f"ff_dense_{i+1}_{j+1}",
                )(ff_output)
                ff_output = layers.Dropout(
                    rate=dropout_rate, name=f"ff_dropout_{i+1}_{j+1}"
                )(ff_output)
            ff_output = layers.Dense(
                hidden_size,
                kernel_regularizer=keras.regularizers.l2(regularization_lambda),
                name=f"ff_dense_final_{i+1}",
            )(ff_output)
            transformer_output = layers.LayerNormalization(
                epsilon=1e-6, name=f"ff_layernorm_{i+1}"
            )(ff_output + attention_output)

        assignment_proportion = int(hidden_size * 0.75)

        # Split the output into regression and assignment outputs
        assignment_head, regression_head = SplitLayer(
            split_size=[assignment_proportion, hidden_size - assignment_proportion],
            axis=-1,
        )(transformer_output)

        # Assignment output
        dim = assignment_proportion
        i = 0
        while dim > 2:
            assignment_head = layers.Dense(
                dim,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(regularization_lambda),
                name=f"assignment_dense_{i+1}",
            )(assignment_head)
            assignment_head = layers.Dropout(
                rate=dropout_rate, name=f"assignment_dropout_{i+1}"
            )(assignment_head)
            dim //= 2
            i += 1
        assignment_head = layers.Dense(
            2,
            activation="sigmoid",
            kernel_regularizer=keras.regularizers.l2(regularization_lambda),
            name="assignment_final_dense",
        )(assignment_head)
        assignment_output = TemporalSoftmax(name="assignment_output", axis=-2)(
            assignment_head, mask=jet_mask
        )

        dim = hidden_size - assignment_proportion
        i = 0
        while dim < hidden_size:
            regression_head = layers.Dense(
                dim,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(regularization_lambda),
                name=f"regression_dense_{i+1}",
            )(regression_head)
            regression_head = layers.Dropout(
                rate=dropout_rate, name=f"regression_dropout_{i+1}"
            )(regression_head)
            dim *= 2
            i += 1

        i += 1
        regression_head = layers.Dense(
            hidden_size,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(regularization_lambda),
            name=f"regression_dense_{i+1}",
        )(regression_head)
        regression_head = layers.Dropout(
            rate=dropout_rate, name=f"regression_dropout_{i+1}"
        )(regression_head)
        regression_head = layers.LayerNormalization(
            epsilon=1e-6, name="regression_layernorm"
        )(regression_head)

        # Another transformer block for regression head
        regression_attention_output = regression_head
        for i in range(transformer_layers):
            # Multi-head self-attention
            regression_attention_output = layers.MultiHeadAttention(
                num_heads=number_attention_heads,
                key_dim=hidden_size,
                dropout=dropout_rate,
                name=f"regression_attention_block_{i+1}",
            )(
                regression_attention_output,
                regression_attention_output,
                regression_attention_output,
                attention_mask=jet_mask,
            )
            regression_attention_output = layers.Dropout(
                rate=dropout_rate, name=f"regression_attention_dropout_{i+1}"
            )(regression_attention_output)
            regression_attention_output = layers.LayerNormalization(
                epsilon=1e-6, name=f"regression_attention_layernorm_{i+1}"
            )(regression_attention_output + regression_head)
            # Residual connection
            regression_attention_output = layers.Dropout(
                rate=dropout_rate, name=f"regression_residual_dropout_{i+1}"
            )(regression_attention_output)
            # Feed-forward network
            ff_output = regression_attention_output
            for j in range(ff_layers):
                ff_output = layers.Dense(
                    hidden_size * 2,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l2(regularization_lambda),
                    name=f"regression_ff_dense_{i+1}_{j+1}",
                )(ff_output)
                ff_output = layers.Dropout(
                    rate=dropout_rate, name=f"regression_ff_dropout_{i+1}_{j+1}"
                )(ff_output)


            regression_attention_output = layers.LayerNormalization(
                epsilon=1e-6, name=f"regression_residual_layernorm_{i+1}"
            )(regression_attention_output + regression_head)

        regression_head = layers.GlobalAveragePooling1D(
            name="regression_global_average_pooling"
        )(regression_head, mask=jet_mask_flatted)


        i = 0
        dim = hidden_size
        while dim > self.n_regression_targets:
            regression_head = layers.Dense(
                dim,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(regularization_lambda),
                name=f"regression_dense_final_{i+1}",
            )(regression_head)
            regression_head = layers.Dropout(
                rate=dropout_rate, name=f"regression_dropout_final_{ff_layers + i+1}"
            )(regression_head)
            dim //= 2
            i += 1
        regression_output = layers.Dense(
            self.n_regression_targets,
            activation="linear",
            kernel_regularizer=keras.regularizers.l2(regularization_lambda),
            name="regression_output",
        )(regression_head)

        # Create the model
        self.model = keras.Model(
            inputs=[jet_input, lepton_input, global_input],
            outputs={"assignment_output": assignment_output, "regression_output": regression_output},
            name="SimpleRegressionModel",
        )


class SPA_NET_Regressor(RegressionBaseModel):
    def build_model(
        self,
        hidden_size=16,
        attention_blocks=3,
        ff_layers=2,
        embedding_layers=2,
        n_heads=4,
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

        # Jet embedding
        jet_embedding = jet_input
        for i in range(embedding_layers):
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
        for i in range(embedding_layers):
            lepton_embedding = layers.Dense(
                hidden_size,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(regularization_lambda),
                name=f"lepton_embedding_{i+1}",
            )(lepton_embedding)
            lepton_embedding = layers.Dropout(
                rate=dropout_rate, name=f"lepton_embedding_dropout_{i+1}"
            )(lepton_embedding)

        # Global embedding
        global_embedding = global_input
        for i in range(embedding_layers):
            global_embedding = layers.Dense(
                hidden_size,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(regularization_lambda),
                name=f"global_embedding_{i+1}",
            )(global_embedding)
            global_embedding = layers.Dropout(
                rate=dropout_rate, name=f"global_embedding_dropout_{i+1}"
            )(global_embedding)

        # Concatenate embeddings
        global_embedding_repeated = layers.Flatten()(global_embedding)
        global_embedding_repeated = layers.RepeatVector(self.max_jets)(
            global_embedding_repeated
        )
        lepton_embedding_repeated = layers.Flatten()(
            lepton_embedding
        )
        lepton_embedding_repeated = layers.RepeatVector(self.max_jets)(
            lepton_embedding_repeated
        )
        concatenated_embeddings = layers.Concatenate(
            axis=-1, name="jet_lepton_global_concat"
        )(
            [jet_embedding, lepton_embedding_repeated, global_embedding_repeated]
        )

        for i in range(embedding_layers):
            concatenated_embeddings = layers.Dense(
                hidden_size,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(regularization_lambda),
                name=f"concatenated_embedding_{i+1}",
            )(concatenated_embeddings)
            concatenated_embeddings = layers.Dropout(
                rate=dropout_rate, name=f"concatenated_embedding_dropout_{i+1}"
            )(concatenated_embeddings)

        event_vector = PrependLearnedVector(
            hidden_size, name="prepend_learned_vector"
        )(concatenated_embeddings)
        concatenated_embeddings = layers.Concatenate(
            axis=-2, name="concatenated_embeddings"
        )([event_vector, concatenated_embeddings])
        transformer_mask = AppendTrueMask(1)(jet_mask)

        # Transformer blocks
        transformer_output = concatenated_embeddings
        for i in range(attention_blocks):
            # Multi-head self-attention
            attention_output = layers.MultiHeadAttention(
                num_heads=n_heads,
                key_dim=hidden_size,
                dropout=dropout_rate,
                name=f"attention_block_{i+1}",
            )(transformer_output, transformer_output, attention_mask=transformer_mask)
            attention_output = layers.Dropout(
                rate=dropout_rate, name=f"attention_dropout_{i+1}"
            )(attention_output)
            attention_output = layers.LayerNormalization(
                epsilon=1e-6, name=f"attention_layernorm_{i+1}"
            )(attention_output + transformer_output)

            # Residual connection
            attention_output = layers.Dropout(
                rate=dropout_rate, name=f"attention_residual_dropout_{i+1}"
            )(attention_output)
            attention_output = layers.LayerNormalization(
                epsilon=1e-6, name=f"attention_residual_layernorm_{i+1}"
            )(attention_output + transformer_output)
            # Feed-forward network
            ff_output = attention_output
            for j in range(ff_layers):
                ff_output = layers.Dense(
                    hidden_size * 2,
                    activation="relu",
                    kernel_regularizer=keras.regularizers.l2(regularization_lambda),
                    name=f"ff_dense_{i+1}_{j+1}",
                )(ff_output)
                ff_output = layers.Dropout(
                    rate=dropout_rate, name=f"ff_dropout_{i+1}_{j+1}"
                )(ff_output)
            ff_output = layers.Dense(
                hidden_size,
                kernel_regularizer=keras.regularizers.l2(regularization_lambda),
                name=f"ff_dense_final_{i+1}",
            )(ff_output)
            transformer_output = layers.LayerNormalization(
                epsilon=1e-6, name=f"ff_layernorm_{i+1}"
            )(ff_output + attention_output)

        jet_output, global_output = SplitLayer(split_size=[self.max_jets, 1])(transformer_output)

        for i in range(embedding_layers):
            jet_output = layers.Dense(
                hidden_size,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(regularization_lambda),
                name=f"jet_output_dense_{i+1}",
            )(jet_output)
            jet_output = layers.Dropout(
                rate=dropout_rate, name=f"jet_output_dropout_{i+1}"
            )(jet_output)
        dim = hidden_size
        i = 0
        while dim > 1:
            jet_output = layers.Dense(
                dim,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(regularization_lambda),
                name=f"jet_output_dense_{embedding_layers+ i+1}",
            )(jet_output)
            jet_output = layers.Dropout(
                rate=dropout_rate, name=f"jet_output_dropout_{embedding_layers+ i+1}"
            )(jet_output)
            dim //= 2
            i += 1

        jet_output = layers.Dense(
            2,
            activation="sigmoid",
            kernel_regularizer=keras.regularizers.l2(regularization_lambda),
            name="jet_output_final",
        )(jet_output)

        assignment_output = TemporalSoftmax(axis=1, name="assignment_output")(
            jet_output, mask=jet_mask
        )

        for i in range(embedding_layers):
            global_output = layers.Dense(
                hidden_size,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(regularization_lambda),
                name=f"global_output_dense_{i+1}",
            )(global_output)
            global_output = layers.Dropout(
                rate=dropout_rate, name=f"global_output_dropout_{i+1}"
            )(global_output)

        dim = hidden_size
        i = 0
        while dim > self.n_regression_targets:
            global_output = layers.Dense(
                dim,
                activation="relu",
                kernel_regularizer=keras.regularizers.l2(regularization_lambda),
                name=f"global_output_dense_{embedding_layers+ i+1}",
            )(global_output)
            global_output = layers.Dropout(
                rate=dropout_rate, name=f"global_output_dropout_{embedding_layers+ i+1}"
            )(global_output)
            dim //= 2
            i += 1
        regression_output = layers.Dense(
            self.n_regression_targets,
            activation="linear",
            kernel_regularizer=keras.regularizers.l2(regularization_lambda),
            name="regression_output",
        )(global_output)

        # Create the model
        self.model = keras.Model(
            inputs=[jet_input, lepton_input, global_input],
            outputs=[assignment_output, regression_output],
            name="SPA_NET_Regressor",
        )
