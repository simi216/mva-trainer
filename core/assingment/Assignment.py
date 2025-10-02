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

        def myprint(s):
            with open(file_path.replace(".keras", "_structure.txt"), "a") as f:
                print(s, file=f)

        self.model.summary(print_fn=myprint)


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

    def predict_indices(self, data_dict, batch_size=512, exclusive = True):
        if self.model is None:
            raise ValueError(
                "Model has not been built yet. Call build_model() before predict_indices()."
            )
        predictions = self.model.predict(
            [data_dict["jet"], data_dict["lepton"], data_dict["global"]], batch_size=batch_size
        )
        if not exclusive:
            return predictions
        else:
            exclusive_predictions = np.zeros_like(predictions)
            for i in range(predictions.shape[0]):
                for lep_idx in range(predictions.shape[2]):
                    flat_idx = np.argmax(predictions[i].flatten())
                    jet_idx = flat_idx // predictions.shape[2]
                    lep_idx = flat_idx % predictions.shape[2]
                    exclusive_predictions[i, jet_idx, lep_idx] = 1
                    predictions[i, jet_idx, :] = -1  # Invalidate this jet for further
                    predictions[i, :, lep_idx] = -1  # Invalidate this lepton for further
            return exclusive_predictions





    def export_to_onnx(self, onnx_file_path="model.onnx"):
        if self.model is None:
            raise ValueError(
                "Model has not been built yet. Call build_model() before export_to_onnx()."
            )
        spec = (tf.TensorSpec((None, self.max_jets, self.n_jets), tf.float32, name="jet_inputs"), 
                tf.TensorSpec((None, self.max_leptons, self.n_leptons), tf.float32, name="lep_inputs"),
                tf.TensorSpec((None, 1, self.n_global), tf.float32, name="global_inputs") if self.n_global > 0 else None)
        model_proto, _ = tf2onnx.convert.from_keras(self.model, input_signature=spec, opset=13)
        onnx.save(model_proto, onnx_file_path)
        print(f"Model exported to {onnx_file_path}")