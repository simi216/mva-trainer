import keras
from keras import regularizers


class MLP(keras.layers.Layer):
    def __init__(
        self,
        output_dim,
        num_layers=3,
        layer_norm=False,
        activation="linear",
        hidden_activation="relu",
        dropout_rate=0,
        regularizer=None,
        name="mlp_dense",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.layer_norm = layer_norm
        self.activation = activation
        self.hidden_activation = hidden_activation
        self.dropout_rate = dropout_rate
        self.regularizer = regularizer
        self.layers_list = []

    def build(self, input_shape):
        input_dim = input_shape[-1]

        if self.num_layers > 1:
            node_list = [
                int(input_dim * ((self.output_dim / input_dim) ** (i / self.num_layers)))
                for i in range(1, self.num_layers)
            ]
        else:
            node_list = []

        prev_dim = input_dim
        for i, width in enumerate(node_list):
            self.layers_list.append(
                keras.layers.Dense(
                    width,
                    activation=self.hidden_activation,
                    name=f"{self.name}_dense_{i}",
                    kernel_regularizer=self.regularizer,
                    bias_regularizer=self.regularizer,
                    activity_regularizer=self.regularizer,
                )
            )
            if self.layer_norm:
                self.layers_list.append(
                    keras.layers.LayerNormalization(name=f"{self.name}_ln_{i}")
                )
            if self.dropout_rate > 0:
                self.layers_list.append(
                    keras.layers.Dropout(self.dropout_rate, name=f"{self.name}_drop_{i}")
                )
            prev_dim = width

        # Final output layer
        self.layers_list.append(
            keras.layers.Dense(
                self.output_dim,
                activation=self.activation,
                name=f"{self.name}_output",
                kernel_regularizer=self.regularizer,
                bias_regularizer=self.regularizer,
                activity_regularizer=self.regularizer,
            )
        )
        if self.layer_norm:
            self.layers_list.append(
                keras.layers.LayerNormalization(name=f"{self.name}_ln_out")
            )
        if self.dropout_rate > 0:
            self.layers_list.append(
                keras.layers.Dropout(self.dropout_rate, name=f"{self.name}_drop_out")
            )

        super().build(input_shape)

    def call(self, inputs, training=None):
        x = inputs
        for layer in self.layers_list:
            if isinstance(layer, keras.layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.output_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "num_layers": self.num_layers,
                "layer_norm": self.layer_norm,
                "activation": self.activation,
                "hidden_activation": self.hidden_activation,
                "dropout_rate": self.dropout_rate,
                "regularizer": regularizers.serialize(self.regularizer),
            }
        )
        return config
