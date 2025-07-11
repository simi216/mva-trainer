from .BaseModel import BaseModel
from .DataLoader import DataPreprocessor
from .CustomObjects import TemporalSoftmax, JetMaskingLayer


import tensorflow as tf
import keras




class RNNJetMatcher(BaseModel):
    def __init__(self, data_preprocessor: DataPreprocessor):
        super().__init__(data_preprocessor)

    def load_data(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test



    def build_model(self,lep_embedding = 4, jet_embedding = 6, lstm_size = 3, *args, **kwargs):
        if self.model is not None:
            raise ValueError(
                "Model already built. Please use a different instance of the class to build a new model."
            )
        jet_input = keras.layers.Input(shape=(self.max_jets, self.n_jets + self.n_combined * self.max_leptons), name="jet_input")
        jet_mask =  JetMaskingLayer(name="jet_mask", padding_value=self.padding_value)(jet_input)
        jet_mask = keras.layers.Reshape(target_shape=(self.max_jets, 1), name="reshaped_jet_mask")(jet_mask)
        lepton_input = keras.layers.Input(shape=((self.max_leptons, self.n_leptons ,)), name="lepton_input")
        lepton_input_reshaped = keras.layers.Reshape(target_shape=(self.n_leptons * self.max_leptons,), name="reshaped_lepton_input")(lepton_input)

        lepton_dnn = keras.layers.Dense(
            lep_embedding,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(1e-4),
            name="lepton_dnn"
        )(lepton_input_reshaped)
        repeated_lepton_input = keras.layers.RepeatVector(self.max_jets, name="repeated_lepton_input")(lepton_dnn)

        jet_dnn = keras.layers.TimeDistributed(
            keras.layers.Dense(
            jet_embedding,
            activation="relu",
            ),
            name="jet_dnn"
        )(jet_input, mask=jet_mask)

        if self.global_features is not None:
            global_input = keras.layers.Input(shape=(1, self.n_global), name="global_input")
            global_input_reshaped = keras.layers.Reshape(target_shape=(self.n_global,), name="reshaped_global_input")(global_input)
            repeated_global_input = keras.layers.RepeatVector(self.max_jets, name="repeated_global_input")(global_input_reshaped)
        else:
            global_input = None

        if self.global_features is not None:
            jet_lepton_concat = keras.layers.Concatenate(axis=-1, name="jet_lepton_concat")([jet_dnn, repeated_lepton_input, repeated_global_input])
        else:
            jet_lepton_concat = keras.layers.Concatenate(axis=-1, name="jet_lepton_concat")([jet_dnn, repeated_lepton_input])

        jet_lepton_embedding = keras.layers.TimeDistributed(keras.layers.Dense(
            9,
            activation="relu",
            name="jet_lepton_embedding"
        ))(jet_lepton_concat, mask=jet_mask)

        #jet_lepton_embedding = keras.layers.Dropout(0.1, name="jet_lepton_embedding_dropout")(jet_lepton_embedding)

        jet_lepton_lstm_1 = keras.layers.Bidirectional(
            keras.layers.LSTM(
            lstm_size,
            return_sequences=True,
            name="jet_lepton_lstm_1"
            )
        )(jet_lepton_embedding, mask=jet_mask)

        jet_lepton_lstm_2 = keras.layers.Bidirectional(
            keras.layers.LSTM(
            lstm_size - 1,
            return_sequences=True,
            name="jet_lepton_lstm_2"
            )
        )(jet_lepton_lstm_1, mask=jet_mask)

        jet_lepton_dnn = keras.layers.TimeDistributed(
            keras.layers.Dense(
            5,
            activation="relu",
            ),
            name="jet_lepton_dnn"
        )(jet_lepton_lstm_2, mask=jet_mask)

        lepton_head_1_dnn = keras.layers.TimeDistributed(
            keras.layers.Dense(
            5,
            activation="relu",
            ),
            name="lepton_head_1_dense"
        )(jet_lepton_dnn, mask=jet_mask)
        lepton_head_1 = keras.layers.TimeDistributed(
            keras.layers.Dense(
            1,
            activation="sigmoid",
            ),
            name="lepton_head_1_output"
        )(lepton_head_1_dnn, mask=jet_mask)
        output_lep_1 = TemporalSoftmax(name="lep_1")(lepton_head_1, mask=jet_mask)

        lepton_head_2_dnn = keras.layers.TimeDistributed(
            keras.layers.Dense(
            5,
            activation="relu",
            ),
            name="lepton_head_2_dense"
        )(jet_lepton_dnn, mask=jet_mask)
        lepton_head_2 = keras.layers.TimeDistributed(
            keras.layers.Dense(
            1,
            activation="sigmoid",
            ),
            name="lepton_head_2_output"
        )(lepton_head_2_dnn, mask=jet_mask)
        output_lep_2 = TemporalSoftmax(name="lep_2")(lepton_head_2, mask=jet_mask)

        combined_output = keras.layers.Concatenate(axis=-1, name="combined_output")([output_lep_1, output_lep_2])

        if self.global_features is not None:
            self.model = keras.Model(
                inputs=[jet_input, lepton_input, global_input],
                outputs=[combined_output],
            )
        else:
            self.model = keras.Model(
                inputs=[jet_input, lepton_input], outputs=[combined_output]
            )
