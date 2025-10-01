import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from core.DataLoader import DataPreprocessor, DataConfig
import core.components as CustomObjects


class JetAssignerBase(ABC):
    def __init__(self, name="jet_assigner"):
        self.name = name

    @abstractmethod
    def predict_indices(self, data_dict):
        pass

    def get_name(self):
        return self.name

class MLAssignerBase(JetAssignerBase):
    def __init__(self, config : DataConfig, name = "ml_assigner"):
        self.name = name
        """
        Initializes the AssignmentBaseModel class.
        Args:
            data_preprocessor (DataPreprocessor): An instance of the DataPreprocessor class
                that provides preprocessed data and metadata required for model initialization.
        """

        config

        self.model: keras.Model = None
        self.X_train = None
        self.X_test: np.ndarray = None
        self.y_train: np.ndarray = None
        self.y_test: np.ndarray = None
        self.history = None
        self.sample_weights = None
        self.class_weights = None
        self.max_leptons = config.max_leptons
        self.max_jets = config.max_jets
        self.global_features = config.global_features
        self.n_jets: int = len(config.jet_features)
        self.n_leptons: int = len(config.lepton_features)
        self.n_global: int = (
            len(config.global_features)
            if config.global_features
            else 0
        )
        self.padding_value: float = config.padding_value
        self.feature_index_dict = config.feature_index_dict

        super().__init__(name=name)

    def load_training_data(self, X_train, y_train, sample_weights=None, class_weights=None):
        self.X_train = X_train
        self.y_train = y_train
        self.sample_weights = sample_weights
        self.class_weights = class_weights

    def compile_model(self, loss, optimizer, metrics=None):
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() before compile_model().")
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def train_model(self, epochs, batch_size, validation_data=None, callbacks=None):
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() before train_model().")
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("Training data not loaded. Call load_training_data() before train_model().")

        if self.history is not None:
            print("Warning: Overwriting existing training history.")

        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            sample_weight=self.sample_weights,
            class_weight=self.class_weights,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
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


    def load_model(self, filepath):
        """
        Loads a Keras model from the specified file path.
        Args:
            filepath (str): The file path from which the model will be loaded.
        Raises:
            ValueError: If the provided file path does not end with ".keras".
            IOError: If the file at the specified path does not exist or cannot be read.
        Side Effects:
            - Loads the model and assigns it to `self.model`.
            - Prints a confirmation message indicating the model has been loaded.
        Example:
            self.load_model("my_model.keras")
        """

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
            for name, obj in CustomObjects.__dict__.items()
            if isinstance(obj, type) and issubclass(obj, keras.layers.Layer)
        }
        custom_objects["CustomUtility>AssignmentLoss"] = CustomObjects.AssignmentLoss
        custom_objects["CustomUtility>accuracy"] = CustomObjects.accuracy
        self.model = keras.saving.load_model(file_path, custom_objects=custom_objects)
        print(f"Model loaded from {file_path}")

    def predict_indices(self, data_dict):
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() before predict_indices().")
        predictions = self.model.predict([data_dict["jet"], data_dict["lepton"], data_dict["global"]])
        return predictions
