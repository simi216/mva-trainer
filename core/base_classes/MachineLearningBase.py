from abc import ABC, abstractmethod
from . import BaseUtilityModel
from core.DataLoader import DataConfig
import keras
import numpy as np
import tensorflow as tf
import tf2onnx
import os
from core.components import onnx_support
from core.components import GenerateMask, InputPtEtaPhiELayer, InputMetPhiLayer
import core.components as components
import core.utils as utils
from copy import deepcopy

@keras.utils.register_keras_serializable()
class KerasModelWrapper(keras.Model):
    def predict_dict(self, x, batch_size=None, verbose=0, steps=None, **kwargs):
        predictions = super().predict(
            x, batch_size=batch_size, verbose=verbose, steps=steps, **kwargs
        )
        if not isinstance(predictions, dict):
            if not isinstance(predictions, list):
                predictions = [predictions]
            return dict(zip(self.output_names, predictions))
        return predictions


class MLWrapperBase(BaseUtilityModel, ABC):
    def __init__(self, config: DataConfig, name="ml_assigner", assignment_name=None, full_reco_name=None):
        """
        Initializes the AssignmentBaseModel class.
        Args:
            data_preprocessor (DataPreprocessor): An instance of the DataPreprocessor class
                that provides preprocessed data and metadata required for model initialization.
        """
        self.model: KerasModelWrapper = None
        self.history = None
        self.sample_weights = None
        self.class_weights = None
        self.NUM_LEPTONS = config.NUM_LEPTONS
        self.max_jets = config.max_jets
        self.met_features = config.met_features
        self.n_jets: int = len(config.jet_features)
        self.n_leptons: int = len(config.lepton_features)
        self.n_met: int = len(config.met_features) if config.met_features else 0
        self.padding_value: float = config.padding_value
        self.feature_index_dict = config.feature_indices
        self.perform_regression = False

        # initialize empty dicts to hold inputs and transformed inputs
        self.inputs = {}
        self.transformed_inputs = {}

        # Use assignment_name and full_reco_name if provided, otherwise fall back to name
        if assignment_name is None:
            assignment_name = name
        if full_reco_name is None:
            full_reco_name = name
        
        super().__init__(config=config, assignment_name=assignment_name, full_reco_name=full_reco_name)

    def build_model(self, **kwargs):
        raise NotImplementedError("Subclasses must implement build_model method.")
        pass

    def prepare_training_data(
        self, X_train, y_train, sample_weights=None, class_weights=None, copy_data=False
    ):
        self.sample_weights = sample_weights
        self.class_weights = class_weights
        if copy_data:
            y_train = deepcopy(y_train)
            X_train = deepcopy(X_train)

        # Rename targets to match model output names
        y_train["assignment"] = y_train.pop("assignment_labels")
        y_train["regression"] = y_train.pop("neutrino_truth")
        if not self.perform_regression:
            y_train.pop("regression")

        jet_data = None
        for key in X_train.keys():
            if "jet" in key:
                jet_data = X_train.pop(key)
                break

        if jet_data is None:
            raise ValueError("Jet data not found in X_train.")
        else:
            X_train["jet_inputs"] = jet_data

        lepton_data = None
        for key in X_train.keys():
            if "lepton" in key:
                lepton_data = X_train.pop(key)
                break
        if lepton_data is None:
            raise ValueError("Lepton data not found in X_train.")
        else:
            X_train["lep_inputs"] = lepton_data
        if self.n_met > 0:
            met_data = None
            for key in X_train.keys():
                if "met" in key:
                    met_data = X_train.pop(key)
                    break
            if met_data is None:
                raise ValueError("met data not found in X_train.")
            else:
                X_train["met_inputs"] = met_data
        return X_train, y_train, sample_weights

    def _prepare_inputs(self, input_as_four_vector, log_E=True):
        jet_inputs = keras.Input(shape=(self.max_jets, self.n_jets), name="jet_inputs")
        lep_inputs = keras.Input(
            shape=(self.NUM_LEPTONS, self.n_leptons), name="lep_inputs"
        )
        met_inputs = keras.Input(shape=(1, self.n_met), name="met_inputs")

        # Normalise inputs
        if input_as_four_vector:
            transformed_jet_inputs = InputPtEtaPhiELayer(
                name="jet_input_transform",
                log_E=log_E,
                padding_value=self.padding_value,
            )(jet_inputs)
            transformed_lep_inputs = InputPtEtaPhiELayer(
                name="lep_input_transform",
                log_E=log_E,
                padding_value=self.padding_value,
            )(lep_inputs)
            transformed_met_inputs = InputMetPhiLayer(name="met_input_transform")(
                met_inputs
            )
        else:
            transformed_jet_inputs = jet_inputs
            transformed_lep_inputs = lep_inputs
            transformed_met_inputs = met_inputs

        normed_jet_inputs = keras.layers.Normalization(name="jet_input_normalization")(
            transformed_jet_inputs
        )
        normed_lep_inputs = keras.layers.Normalization(name="lep_input_normalization")(
            transformed_lep_inputs
        )
        normed_met_inputs = keras.layers.Normalization(name="met_input_normalization")(
            transformed_met_inputs
        )
        # Generate masks
        jet_mask = GenerateMask(padding_value=-999, name="jet_mask")(jet_inputs)

        self.inputs = {
            "jet_inputs": jet_inputs,
            "lep_inputs": lep_inputs,
            "met_inputs": met_inputs,
        }
        self.transformed_inputs = {
            "jet_inputs": transformed_jet_inputs,
            "lepton_inputs": transformed_lep_inputs,
            "met_inputs": transformed_met_inputs,
        }
        return normed_jet_inputs, normed_lep_inputs, normed_met_inputs, jet_mask

    def train_model(
        self,
        X_train,
        y_train,
        epochs,
        batch_size,
        sample_weights=None,
        validation_split=0.2,
        callbacks=None,
        copy_data = False,
        **kwargs,
    ):
        if self.model is None:
            raise ValueError(
                "Model has not been built yet. Call build_model() before train_model()."
            )

        X_train, y_train, sample_weights = self.prepare_training_data(
            X_train, y_train, sample_weights=sample_weights, copy_data=copy_data
        )

        if self.history is not None:
            print("Warning: Overwriting existing training history.")

        self.history = self.model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
            class_weight=self.class_weights,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            **kwargs,
        )
        return self.history

    def save_model(self, file_path="model.keras"):
        """
        Saves the current model to the specified file path in Keras format and writes the model's structure to a text file.
        Args:
            file_path (str): The file path where the model will be saved.
                            Must end with ".keras". Defaults to "model.keras".
        Raises:
            ValueError: If the model has not been built (i.e., `self.model` is None).
            ValueError: If the provided file path does not end with ".keras".
        Side Effects:
            - Saves the model to the specified file path in Keras format.
            - Writes the model's structure to a text file with the same name as the model file,
            but with "_structure.txt" appended instead of ".keras".
            - Prints a confirmation message indicating the model has been saved.
        Example:
            self.save_model("my_model.keras")
        """

        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )
        if ".keras" not in file_path:
            raise ValueError(
                "File path must end with .keras. Please provide a valid file path."
            )
        self.model.save(file_path)

        if self.history is not None:
            history_path = file_path.replace(".keras", "_history")
            np.savez(history_path, **self.history.history)
        print(f"Model saved to {file_path}")


    def load_model(self, file_path):
        """
        Loads a pre-trained Keras model from the specified file path.

        This method uses custom objects defined in the `CustomObjects` class to
        ensure that any custom layers, loss functions, or metrics used in the
        model are properly loaded.

        Args:
            file_path (str): The file path to the saved Keras model.

        Attributes:
            self.model: The loaded Keras model instance.

        Raises:
            ImportError: If the required Keras modules are not available.
            ValueError: If the file_path is invalid or the model cannot be loaded.

        Example:
            >>> model_instance = AssignmentBaseModel()
            >>> model_instance.load_model("/path/to/saved_model")
            Model loaded from /path/to/saved_model
        """

        custom_objects = {
            name: obj
            for name, obj in zip(components.__dict__.items(), utils.__dict__.items())
            if isinstance(obj, type) and issubclass(obj, keras.layers.Layer)
        }
        custom_objects.update({"KerasModelWrapper": KerasModelWrapper})

        self.model = keras.saving.load_model(file_path, custom_objects=custom_objects)
        print(f"Model loaded from {file_path}")
        history_path = file_path.replace(".keras", "_history.npz")
        if os.path.exists(history_path):
            loaded_history = np.load(history_path, allow_pickle=True)
            history_dict = {key: loaded_history[key].tolist() for key in loaded_history}
            self.history = keras.callbacks.History()
            self.history.history = history_dict
            print(f"Training history loaded from {history_path}")
        else:
            print(f"WARNING: No training history found at {history_path}")


    def adapt_normalization_layers(self, data: dict):
        """
        Adapts the normalization layers in a functional model.
        Each normalization layer is adapted using data that has been passed
        through all preceding layers (up to but excluding the normalization layer).
        """
        # --- Prepare and unpad jet data ---
        jet_data = data["jet"]  # (num_events, n_jets, n_features)
        jet_mask = np.any(jet_data != self.padding_value, axis=-1)
        unpadded_jet_data = jet_data[jet_mask]
        num_jets = unpadded_jet_data.shape[0]
        num_events = num_jets // self.max_jets
        unpadded_jet_data = unpadded_jet_data[: num_events * self.max_jets, :].reshape(
            (num_events, self.max_jets, self.n_jets)
        )
        lep_data = data["lepton"][:num_events, :, :]
        met_data = data["met"][:num_events, :, :]

        # --- Helper: build a submodel up to (but not including) a target layer ---
        def get_pre_norm_submodel(model, target_layer_name):
            target_layer = model.get_layer(target_layer_name)
            # find which input(s) feed into this layer
            inbound_nodes = target_layer._inbound_nodes
            if not inbound_nodes:
                raise ValueError(f"Layer '{target_layer_name}' has no inbound nodes.")
            # assume single inbound node (standard case)
            inbound_tensors = inbound_nodes[0].input_tensors
            # find model input(s) corresponding to those tensors
            # Build a submodel from inputs -> pre-normalization outputs
            submodel = keras.Model(
                inputs=model.inputs,
                outputs=(
                    inbound_tensors if len(inbound_tensors) > 1 else inbound_tensors[0]
                ),
                name=f"pre_{target_layer_name}_model",
            )
            return submodel

        # --- Loop over normalization layers and adapt each ---
        for layer in self.model.layers:
            if isinstance(layer, keras.layers.Normalization):
                submodel = get_pre_norm_submodel(self.model, layer.name)

                if layer.name == "jet_input_normalization":
                    transformed = submodel(
                        {
                            "jet_inputs": unpadded_jet_data,
                            "lep_inputs": lep_data,
                            "met_inputs": met_data,
                        }
                    )
                    layer.adapt(transformed)
                elif layer.name == "lep_input_normalization":
                    transformed = submodel(
                        {
                            "jet_inputs": unpadded_jet_data,
                            "lep_inputs": lep_data,
                            "met_inputs": met_data,
                        }
                    )
                    layer.adapt(transformed)
                elif layer.name == "met_input_normalization":
                    transformed = submodel(
                        {
                            "jet_inputs": unpadded_jet_data,
                            "lep_inputs": lep_data,
                            "met_inputs": met_data,
                        }
                    )
                    layer.adapt(transformed)

    def export_to_onnx(self, onnx_file_path="model.onnx"):
        """
        Exports the current Keras model to onnx format.

        The model is wrapped to take a flattened input tensor and split it into the original inputs.

        Saves a .txt file with model input names and positions for reference.

        Args:
            onnx_file_path (str): The file path where the ONNX model will be saved.
                                  Must end with ".onnx". Defaults to "model.onnx".
        Raises:
            ValueError: If the model has not been built (i.e., `self.model` is None).
            ValueError: If the provided file path does not end with ".onnx".
        """
        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )
        if ".onnx" not in onnx_file_path:
            raise ValueError(
                "File path must end with .onnx. Please provide a valid file path."
            )

        # Define input shapes
        jet_shape = (self.max_jets, self.n_jets)
        lep_shape = (self.NUM_LEPTONS, self.n_leptons)
        input_shapes = [jet_shape, lep_shape]
        if self.n_met > 0:
            met_shape = (1, self.n_met)
            input_shapes.append(met_shape)

        # Create a new model that takes a flat input and splits it
        flat_input_size = sum(np.prod(shape) for shape in input_shapes)
        flat_input = keras.Input(shape=(flat_input_size,), name="flat_input")

        split_layer = onnx_support.SplitInputsLayer(input_shapes)
        split_inputs = split_layer(flat_input)

        if self.n_met > 0:
            jet_input, lep_input, met_input = split_inputs

            model_outputs = self.model([jet_input, lep_input, met_input])
        else:
            jet_input, lep_input = split_inputs
            model_outputs = self.model([jet_input, lep_input])

        wrapped_model = keras.Model(inputs=flat_input, outputs=model_outputs)

        # Convert to ONNX
        spec = (tf.TensorSpec((None, flat_input_size), tf.float32, name="flat_input"),)
        onnx_model, _ = tf2onnx.convert.from_keras(
            wrapped_model,
            input_signature=spec,
            opset=13,
            output_path=onnx_file_path,
        )
        print(f"ONNX model saved to {onnx_file_path}")

        # Save input names and positions to a text file
        input_info_path = onnx_file_path.replace(".onnx", "_input_info.txt")
        feature_index_dict = self.config.feature_indices
        flat_feature_index_dict = {}

        for feature_type, features in feature_index_dict.items():
            if feature_type == "jet":
                for i, feature in enumerate(features):
                    for j in range(self.max_jets):
                        flat_index = j * self.n_jets + i
                        flat_feature_index_dict[flat_index] = f"jet_{j}_{feature}"
            elif feature_type == "lepton":
                for i, feature in enumerate(features):
                    for j in range(self.NUM_LEPTONS):
                        flat_index = (
                            self.max_jets * self.n_jets + j * self.n_leptons + i
                        )
                        flat_feature_index_dict[flat_index] = f"lepton_{j}_{feature}"
            elif feature_type == "met":
                for i, feature in enumerate(features):
                    flat_index = (
                        self.max_jets * self.n_jets
                        + self.NUM_LEPTONS * self.n_leptons
                        + i
                    )
                    flat_feature_index_dict[flat_index] = f"met_{feature}"

        inputs_list = [None] * flat_input_size
        for idx, name in flat_feature_index_dict.items():
            inputs_list[idx] = name

    def print_TopCPToolkitConfig(self, output_file):
        """
        Prints a configuration file for TopCPToolKit based on the model's feature indices.
        Args:
            output_file (str): The file path where the configuration will be saved.
        """
        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )
        with open(output_file, "w") as f:
            f.write("DiLepAssigner\n")
            f.write("\tjets: ")
            f.write("\telectrons: ")
            f.write("\tmuons: ")
            f.write("\tmet: ")
            f.write('\tbtagger: "GN2v01"')
            f.write("\teventSelection: \n")
            f.write("\tn_jets: {}\n".format(self.max_jets))
            f.write("\tNN_padding_value: {}\n".format(self.padding_value))
            f.write("\tmodel_paths: {}\n".format("path/to/model"))
            f.write("\n")
        print(f"TopCPToolKit configuration saved to {output_file}")
