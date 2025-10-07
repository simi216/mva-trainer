import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from core.DataLoader import DataPreprocessor, DataConfig
import core.components as components
import core.utils as utils
import tf2onnx
import onnx
from core.components import onnx_support

class JetAssignerBase(ABC):
    def __init__(self,config : DataConfig, name="jet_assigner"):
        self.name = name
        self.config = config

    @abstractmethod
    def predict_indices(self, data_dict):
        pass

    def get_name(self):
        return self.name

    def evaluate_accuracy(self, data_dict, true_labels, per_event=False):
        """
        Evaluates the model's performance on the provided data and true indices.

        Args:
            data_dict (dict): A dictionary containing input data for the model.
            true_labels (np.ndarray): The true labels (one-hot) to compare against the model's predictions.
        Returns:
            float | np.ndarray: The accuracy of the model's predictions. If per_event is True,
            returns an array of booleans indicating correctness per event; otherwise, returns overall accuracy.
        """
        predictions = self.predict_indices(data_dict)
        predicted_indices = np.argmax(predictions, axis=-2)
        true_indices = np.argmax(true_labels, axis=-2)
        if per_event:
            correct_predictions = np.all(predicted_indices == true_indices, axis=-1)
        else:
            correct_predictions = np.all(predicted_indices == true_indices, axis=-1)
            accuracy = np.mean(correct_predictions)
        return accuracy

class MLAssignerBase(JetAssignerBase):
    def __init__(self, config: DataConfig, name="ml_assigner"):
        self.name = name
        """
        Initializes the AssignmentBaseModel class.
        Args:
            data_preprocessor (DataPreprocessor): An instance of the DataPreprocessor class
                that provides preprocessed data and metadata required for model initialization.
        """

        self.config = config

        self.model: keras.Model = None
        self.X_train = None
        self.y_train: np.ndarray = None
        self.history = None
        self.sample_weights = None
        self.class_weights = None
        self.max_leptons = config.max_leptons
        self.max_jets = config.max_jets
        self.global_features = config.global_features
        self.n_jets: int = len(config.jet_features)
        self.n_leptons: int = len(config.lepton_features)
        self.n_global: int = (
            len(config.global_features) if config.global_features else 0
        )
        self.padding_value: float = config.padding_value
        self.feature_index_dict = config.get_feature_index_dict()

        super().__init__(config=config,name=name)

    def load_training_data(
        self, X_train, y_train, sample_weights=None, class_weights=None
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.sample_weights = sample_weights
        self.class_weights = class_weights

        jet_data = None
        for key in self.X_train.keys():
            if "jet" in key:
                jet_data = self.X_train[key]
                break

        if jet_data is None:
            raise ValueError("Jet data not found in X_train.")
        else:
            self.X_train["jet_inputs"] = jet_data

        lepton_data = None
        for key in self.X_train.keys():
            if "lepton" in key:
                lepton_data = self.X_train[key]
                break
        if lepton_data is None:
            raise ValueError("Lepton data not found in X_train.")
        else:
            self.X_train["lep_inputs"] = lepton_data
        if self.n_global > 0:
            global_data = None
            for key in self.X_train.keys():
                if "global" in key:
                    global_data = self.X_train[key]
                    break
            if global_data is None:
                raise ValueError("Global data not found in X_train.")
            else:
                self.X_train["global_inputs"] = global_data

    def compile_model(self, loss, optimizer, metrics=None):
        if self.model is None:
            raise ValueError(
                "Model has not been built yet. Call build_model() before compile_model()."
            )
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def train_model(self, X_train, y_train, epochs, batch_size, sample_weights = None, validation_split=0.2, callbacks=None):
        if self.model is None:
            raise ValueError(
                "Model has not been built yet. Call build_model() before train_model()."
            )

        self.load_training_data(X_train, y_train, sample_weights = sample_weights)

        if self.history is not None:
            print("Warning: Overwriting existing training history.")

        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            sample_weight=self.sample_weights,
            class_weight=self.class_weights,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
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

    def save_structure(self, file_path="model_structure.txt"):
        """
        Saves the model's structure to a text file.

        This method writes the summary of the Keras model to a specified text file.
        It is useful for documenting the architecture of the model without saving
        the entire model weights and configuration.

        Args:
            file_path (str): The file path where the model structure will be saved.
                             Defaults to "model_structure.txt".

        Raises:
            ValueError: If the model has not been built (i.e., `self.model` is None).

        Side Effects:
            - Writes the model's structure to the specified text file.
            - Prints a confirmation message indicating the structure has been saved.

        Example:
            self.save_structure("my_model_structure.txt")
        """

        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )

        def myprint(s):
            with open(file_path, "a") as f:
                print(s, file=f)

        self.model.summary(print_fn=myprint)
        print(f"Model structure saved to {file_path}")


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

        self.model = keras.saving.load_model(file_path, custom_objects=custom_objects)
        print(f"Model loaded from {file_path}")

    def predict_indices(self, data : dict[str:np.ndarray], exclusive=True):
        """
        Predicts the indices of jets and leptons based on the model's predictions.
        This method processes the predictions from the model and returns a one-hot
        encoded array indicating the associations between jets and leptons.
        Args:
            data (dict): A dictionary containing input data for prediction. It should
                include keys "jet" and "lepton", and optionally "global" if global
                features are used by the model.
            exclusive (bool, optional): If True, ensures exclusive assignments between
                jets and leptons, where each jet is assigned to at most one lepton and
                vice versa. Defaults to True.
        Returns:
            np.ndarray: A one-hot encoded array of shape (batch_size, max_jets, 2),
            where the last dimension represents the association between jets and
            leptons. The value 1 indicates an association, and 0 indicates no association.
        Raises:
            ValueError: If the model is not built (i.e., `self.model` is None).
        Notes:
            - If `exclusive` is True, the method ensures that each jet is assigned
              to at most one lepton and each lepton is assigned to at most one jet.
            - If `exclusive` is False, the method assigns jets and leptons based on
              the maximum prediction probabilities independently for each class.
        """

        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )
        if self.global_features is not None:
            predictions = self.model.predict(
                [data["jet"], data["lepton"], data["global"]], verbose=0
            )
        else:
            predictions = self.model.predict([data["jet"], data["lepton"]], verbose=0)
        one_hot = self.generate_one_hot_encoding(predictions, exclusive)
        return one_hot

    def generate_one_hot_encoding(self, predictions, exclusive):
        """
        Generates a one-hot encoded array from the model's predictions.
        This method processes the raw predictions from the model and converts them
        into a one-hot encoded format, indicating the associations between jets and leptons.
        Args:
            predictions (np.ndarray): The raw predictions from the model, typically
                of shape (batch_size, max_jets, max_leptons).
            exclusive (bool): If True, ensures exclusive assignments between jets
                and leptons
        Returns:
            np.ndarray: A one-hot encoded array of shape (batch_size, max_jets, 2),
            where the last dimension represents the association between jets and
            leptons. The value 1 indicates an association, and 0 indicates no association.
        """
        if exclusive:
            one_hot = np.zeros((predictions.shape[0], self.max_jets, 2), dtype=int)
            for i in range(predictions.shape[0]):
                probs = predictions[i].copy()
                for _ in range(self.max_leptons):
                    jet_index, lepton_index = np.unravel_index(
                        np.argmax(probs), probs.shape
                    )
                    one_hot[i, jet_index, lepton_index] = 1
                    probs[jet_index, :] = 0
                    probs[:, lepton_index] = 0
        else:
            one_hot[
                np.arange(predictions.shape[0]), np.argmax(predictions[:, :, 0], axis=1), 0
            ] = 1
            one_hot[
                np.arange(predictions.shape[0]), np.argmax(predictions[:, :, 1], axis=1), 1
            ] = 1
        return one_hot

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
        lep_shape = (self.max_leptons, self.n_leptons)
        input_shapes = [jet_shape, lep_shape]
        if self.n_global > 0:
            global_shape = (1,self.n_global)
            input_shapes.append(global_shape)

        # Create a new model that takes a flat input and splits it
        flat_input_size = sum(np.prod(shape) for shape in input_shapes)
        flat_input = keras.Input(shape=(flat_input_size,), name="flat_input")

        split_layer = onnx_support.SplitInputsLayer(input_shapes)
        split_inputs = split_layer(flat_input)

        if self.n_global > 0:
            jet_input, lep_input, global_input = split_inputs
            
            model_outputs = self.model([jet_input, lep_input, global_input])
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
        feature_index_dict = self.config.get_feature_index_dict()
        flat_feature_index_dict = {}

        for feature_type, features in feature_index_dict.items():
            if feature_type == "jet":
                for i, feature in enumerate(features):
                    for j in range(self.max_jets):
                        flat_index = j * self.n_jets + i
                        flat_feature_index_dict[flat_index] = f"jet_{j}_{feature}"
            elif feature_type == "lepton":
                for i, feature in enumerate(features):
                    for j in range(self.max_leptons):
                        flat_index = (
                            self.max_jets * self.n_jets + j * self.n_leptons + i
                        )
                        flat_feature_index_dict[flat_index] = f"lepton_{j}_{feature}"
            elif feature_type == "global":
                for i, feature in enumerate(features):
                    flat_index = self.max_jets * self.n_jets + self.max_leptons * self.n_leptons + i
                    flat_feature_index_dict[flat_index] = f"global_{feature}"
    
        inputs_list = [None] * flat_input_size
        for idx, name in flat_feature_index_dict.items():
            inputs_list[idx] = name


        with open(input_info_path, "w") as f:
            f.write("Flat Input Index Mapping:\n")
            for idx, feature in enumerate(inputs_list):
                f.write(f"Index {idx}: {feature}\n")
        print(f"Input info saved to {input_info_path}")