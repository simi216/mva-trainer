import tensorflow as tf
import numpy as np
from .DataLoader import DataLoader, DataPreprocessor
from . import CustomObjects

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

import tensorflow as tf
import keras

import sklearn as sk

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow import convert_to_tensor


class AssignmentBaseModel:
    """
    AssignmentBaseModel is a base class for building and training machine learning models
    for jet-lepton assignment tasks. It provides methods for data loading, model compilation,
    training, evaluation, and visualization.
    Attributes:
        model (keras.Model): The Keras model to be built and trained.
        X_train (dict or np.ndarray): Training features.
        X_test (dict or np.ndarray): Testing features.
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Testing labels.
        history (keras.callbacks.History): Training history.
        sample_weights (np.ndarray): Sample weights for training.
        class_weights (dict): Class weights for training.
        max_leptons (int): Maximum number of leptons.
        max_jets (int): Maximum number of jets.
        global_features (list): List of global features.
        n_jets (int): Number of jet features.
        n_leptons (int): Number of lepton features.
        n_global (int): Number of global features.
        padding_value (float): Padding value for missing data.
        combined_features (list): Combined features for training.
        n_combined (int): Number of combined features.
        feature_index_dict (dict): Dictionary mapping feature names to indices.
        feature_data (dict): Dictionary containing feature data.
    Methods:
        load_data(X_train, y_train, X_test, y_test):
            Load the training and testing data into the model.
        compile_model(lambda_excl=0.5, *args, **kwargs):
            Compile the model with a custom loss function and optional arguments.
        build_model():
            Abstract method to be implemented in subclasses for building the model.
        summary():
            Print the summary of the model.
        train_model(epochs=10, batch_size=32, weight=None, *args, **kwargs):
            Train the model with the provided data and optional weights.
        save_model(file_path):
            Save the model to the specified file path.
        load_model(file_path):
            Load a model from the specified file path.
        predict(data):
            Make predictions using the trained model.
        get_model_summary():
            Get the summary of the model.
        plot_history():
            Plot the training history of the model.
        get_test_data():
            Retrieve the test data.
        compute_sample_weights():
            Compute sample weights for training based on class distribution.
        enhance_region(variable, data_type, low_cut=None, high_cut=None, factor=1.0):
            Enhance a specific region of the data by modifying sample weights.
        predict_indices(data, exclusive=True):
            Predict indices for jet-lepton assignments.
        duplicate_jets():
            Compute the fraction of duplicate jet assignments.
        compute_permutation_importance(shuffle_number=1):
            Compute permutation importance for features.
        plot_permutation_importance(shuffle_number, file_name=None):
            Plot permutation importance for features.
        accuracy_vs_feature(feature_name, data_type="non_training", bins=50, xlims=None):
            Plot accuracy versus a specific feature.
        evaluate_accuracy(data=None, predictions=None):
            Evaluate the accuracy of the model.
        compute_accuracy(truth, prediction):
            Compute accuracy metrics for predictions.
        get_binned_accuracy(feature_name, data_type="non_training", bins=20, xlims=None):
            Compute binned accuracy for a specific feature.
        compute_binned_accuracy(truth, prediction, feature_data, bins=20, xlims=None, event_weights=None):
        plot_accuracy_feature(truth, prediction, feature_data, feature_name, bins=20, xlims=None, fig=None, ax=None, accuracy_color="tab:blue", label=None, event_weights=None):
            Plot accuracy versus a feature.
        plot_external_confusion_matrix(jet_matcher_indices, truth_index, max_jets, n_bootstrap=100, sample_weight=None):
            Plot confusion matrix with bootstrapping.
        plot_confusion_matrix(n_bootstrap=100, exclusive=True):
            Plot confusion matrix for the model's predictions.
        get_confusion_matrix(data=None, labels=None, sample_weights=None, exclusive=True):
            Compute the confusion matrix for the model's predictions.
    """

    def __init__(self, data_preprocessor: DataPreprocessor):
        """
        Initializes the AssignmentBaseModel class.
        Args:
            data_preprocessor (DataPreprocessor): An instance of the DataPreprocessor class
                that provides preprocessed data and metadata required for model initialization.
        Attributes:
            model (keras.Model): The Keras model instance. Defaults to None.
            X_train: Training feature data. Defaults to None.
            X_test (np.ndarray): Testing feature data. Defaults to None.
            y_train (np.ndarray): Training labels. Defaults to None.
            y_test (np.ndarray): Testing labels. Defaults to None.
            history: Training history of the model. Defaults to None.
            sample_weights: Sample weights for training. Defaults to None.
            class_weights: Class weights for training. Defaults to None.
            max_leptons (int): Maximum number of leptons in the dataset.
            max_jets (int): Maximum number of jets in the dataset.
            global_features (list): List of global features in the dataset.
            n_jets (int): Number of jet features.
            n_leptons (int): Number of lepton features.
            n_global (int): Number of global features. Defaults to 0 if no global features are provided.
            padding_value (float): Padding value used in the dataset.
            combined_features (list): List of combined features in the dataset.
            n_combined (int): Number of combined features.
            feature_index_dict (dict[str, any]): Dictionary mapping feature names to their indices.
            feature_data (dict[str, np.ndarray]): Dictionary containing feature data. Defaults to None.
        """

        self.model: keras.Model = None
        self.X_train = None
        self.X_test: np.ndarray = None
        self.y_train: np.ndarray = None
        self.y_test: np.ndarray = None
        self.history = None
        self.sample_weights = None
        self.class_weights = None
        self.max_leptons = data_preprocessor.max_leptons
        self.max_jets = data_preprocessor.max_jets
        self.global_features = data_preprocessor.global_features
        self.n_jets: int = len(data_preprocessor.jet_features)
        self.n_leptons: int = len(data_preprocessor.lepton_features)
        self.n_global: int = (
            len(data_preprocessor.global_features)
            if data_preprocessor.global_features
            else 0
        )
        self.padding_value: float = data_preprocessor.padding_value
        self.combined_features = data_preprocessor.combined_features
        self.n_combined: int = data_preprocessor.n_combined
        self.feature_index_dict: dict[str:any] = data_preprocessor.feature_index_dict
        # self.data_preprocessor = data_preprocessor
        self.feature_data: dict[str : np.ndarray] = None

    def load_data(self, X_train=None, y_train=None, X_test=None, y_test=None):
        """
        Load the training and testing data into the model.
        Parameters:
        ----------
        X_train : array-like, optional
            Training feature data. Default is None.
        y_train : array-like, optional
            Training target labels. Default is None.
        X_test : array-like, optional
            Testing feature data. Default is None.
        y_test : array-like, optional
            Testing target labels. Default is None.
        Notes:
        -----
        If all parameters (X_train, y_train, X_test, y_test) are provided,
        they will be assigned to the corresponding attributes of the model.
        Otherwise, the method assumes that the data will be loaded using. Which is for e
        """
        if (
            X_train is not None
            and X_test is not None
            and y_train is not None
            and y_test is not None
        ):
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test

    def compile_model(self, lambda_excl=0.5, *args, **kwargs):
        """
        Compile the model with a custom loss function and optional arguments.
        Parameters:
        ----------
        lambda_excl : float, optional
            Exclusion penalty for the custom loss function. Default is 0.5.
        *args : tuple, optional
            Additional positional arguments for the compile method.
        **kwargs : dict, optional
            Additional keyword arguments for the compile method.
        Notes:
        -----
        This method compiles the model using a custom loss function defined in CustomObjects.AssignmentLoss.
        The loss function incorporates an exclusion penalty to handle the assignment task effectively.
        """
        self.model.compile(
            loss=CustomObjects.AssignmentLoss(lambda_excl=lambda_excl),
            metrics=["accuracy"],
            *args,
            **kwargs,
        )

    def build_model(self):
        """
        Abstract method to be implemented in subclasses for building the model.
        Raises:
        -------
        NotImplementedError: If this method is not implemented in a subclass.
        """
        raise NotImplementedError(
            "build_model() method must be implemented in subclasses."
        )

    def summary(self):
        """
        Print the summary of the model.
        Returns:
        -------
        str: Summary of the model.
        Raises:
        -------
        ValueError: If the model is not built yet.
        Notes:
        -----
        This method returns the summary of the model architecture.
        It raises a ValueError if the model has not been built yet.
        """
        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )
        return self.model.summary()

    def train_model(self, epochs=10, batch_size=32, weight=None, *args, **kwargs):
        """
        Trains the model using the provided training data and parameters.
        Args:
            epochs (int, optional): Number of epochs to train the model. Defaults to 10.
            batch_size (int, optional): Number of samples per gradient update. Defaults to 32.
            weight (str, optional): Type of weights to use during training.
                Can be 'sample' for sample weights or 'class' for class weights. Defaults to None.
            *args: Additional positional arguments to pass to the `fit` method of the model.
            **kwargs: Additional keyword arguments to pass to the `fit` method of the model.
        Raises:
            ValueError: If the model is not built using the `build_model` method.
            ValueError: If the training data (`X_train` or `y_train`) is not loaded.
            ValueError: If an invalid weight type is provided (not 'sample' or 'class').
        Notes:
            - If `weight` is set to 'sample', sample weights are computed using `compute_sample_weights`.
            - If `weight` is set to 'class', class weights are computed using `compute_class_weights`.
            - If `global_features` are available, they are included in the input data for training.
        Attributes:
            history: The history object returned by the `fit` method, containing details of the training process.
        Example:
            >>> model.train_model(epochs=20, batch_size=64, weight='class')
        """
        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not loaded. Please prepare data using method.")
        print("Starting training...")
        if weight is not None:
            if weight == "sample":
                self.compute_sample_weights()
            elif weight == "class":
                self.compute_class_weights()
            else:
                raise ValueError("Invalid weight type. Use 'sample' or 'class'.")

        if self.class_weights is not None:
            kwargs["class_weight"] = self.class_weights
        elif self.sample_weights is not None:
            kwargs["sample_weight"] = self.sample_weights

        if self.global_features is not None:
            inputs = [
                self.X_train["jet"],
                self.X_train["lepton"],
                self.X_train["global"],
            ]
            validation_data = (
                [self.X_test["jet"], self.X_test["lepton"], self.X_test["global"]],
                (self.y_test),
            )
        else:
            inputs = [self.X_train["jet"], self.X_train["lepton"]]
            validation_data = (
                [self.X_test["jet"], self.X_test["lepton"]],
                (self.y_test),
            )
        self.history = self.model.fit(
            inputs,
            (self.y_train),
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            *args,
            **kwargs,
        )

        print("Training completed.")

    def save_model(self, file_path):
        """
        Saves the built model to the specified file path.
        Args:
            file_path (str): The path where the model will be saved. Must end with '.keras'.
        Raises:
            ValueError: If the model is not built yet or if the file path does not end with '.keras'.
        Notes:
            This method saves the model architecture and weights to a file in Keras format.
            The file can be loaded later using the `load_model` method.
        """
        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )

        self.model.save(file_path)
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
            for name, obj in CustomObjects.__dict__.items()
            if isinstance(obj, type) and issubclass(obj, keras.layers.Layer)
        }
        custom_objects["CustomUtility>AssignmentLoss"] = CustomObjects.AssignmentLoss
        custom_objects["CustomUtility>accuracy"] = CustomObjects.accuracy
        self.model = keras.saving.load_model(file_path, custom_objects=custom_objects)
        print(f"Model loaded from {file_path}")

    def predict(self, data):
        """
        Generate predictions for the given input data using the built model.
        Args:
            data (array-like): Input data for which predictions are to be made.
                               The format and shape of the data should match the
                               requirements of the model.
        Returns:
            array-like: Predictions generated by the model for the input data.
        Raises:
            ValueError: If the model has not been built prior to calling this method.
        """

        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )

        predictions = self.model.predict(data, verbose=0)
        return predictions

    def get_model_summary(self):
        """
        Provides a summary of the model architecture.
        This method returns a summary of the model, which includes details about
        the layers, output shapes, and the number of parameters. If the model has
        not been built yet, it raises a ValueError.
        Returns:
            str: A summary of the model architecture.
        Raises:
            ValueError: If the model has not been built using the build_model() method.
        """

        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )

        return self.model.summary()

    def plot_history(self):
        """
        Plots the training and validation loss and accuracy over epochs.
        This method visualizes the training history of the model, including the
        loss and accuracy for both the training and validation datasets. It
        requires that the model has been trained and the `self.history` attribute
        is populated.
        Raises:
            ValueError: If `self.history` is None, indicating that the model has
                        not been trained yet.
        Returns:
            tuple: A tuple containing the matplotlib figure and axes objects
                   (`fig`, `ax`) for further customization or saving.
        """

        if self.history is None:
            raise ValueError(
                "No training history available. Please train the model using train_model() method."
            )

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(self.history.history["loss"], label="train_loss")
        ax[0].plot(self.history.history["val_loss"], label="val_loss")
        ax[0].set_title("Loss")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Loss")
        ax[0].legend()
        ax[1].plot(self.history.history["accuracy"], label="train_accuracy")
        ax[1].plot(self.history.history["val_accuracy"], label="val_accuracy")
        ax[1].set_title("Accuracy")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Accuracy")
        ax[1].legend()
        return fig, ax

    def get_test_data(self):
        """
        Retrieve the test data (features and labels).
        This method returns the test dataset, which consists of the features (X_test)
        and the corresponding labels (y_test). If the data has not been split using
        the `split_data()` method, a ValueError is raised.
        Returns:
            tuple: A tuple containing X_test (test features) and y_test (test labels).
        Raises:
            ValueError: If the data has not been split using the `split_data()` method.
        """

        if self.X_test is None or self.y_test is None:
            raise ValueError(
                "Data not split. Please split data using split_data() method."
            )

        return self.X_test, self.y_test

    def compute_sample_weights(self):
        """
        Computes sample weights for training data based on the number of jets, class distribution,
        and optional event weights.
        This method calculates weights for each training sample to balance the dataset
        and account for class imbalances or other weighting factors. The weights are
        normalized and optionally combined with event-specific weights.
        Steps:
        1. Initializes sample weights to ones.
        2. Computes the number of jets for each sample.
        3. Assigns weights inversely proportional to the number of samples for each jet count.
        4. Normalizes the weights to have a mean of 1.
        5. Multiplies the weights by event-specific weights if provided.
        6. Updates or initializes the `self.sample_weights` attribute.
        Attributes:
            self.y_train (np.ndarray): The training labels, where each row corresponds to a sample.
            self.padding_value (any): The value used to pad the training labels.
            self.X_train (dict): A dictionary containing training data, including optional
                                 "event_weight" for each sample.
            self.sample_weights (np.ndarray or None): Existing sample weights to be updated,
                                                      or None if not previously defined.
        Returns:
            None: Updates the `self.sample_weights` attribute in place.
        """

        sample_weights = np.ones(self.y_train.shape[0])
        classes = np.unique(self.y_train, axis=0)
        jet_numbers = np.sum(
            ~np.all(self.y_train == self.padding_value, axis=(-1)), axis=-1
        )

        for i, jet_number in enumerate(np.unique(jet_numbers)):
            jet_indices = np.where(jet_numbers == jet_number)[0]
            if len(jet_indices) > 0:
                class_weight = 1.0 / len(jet_indices) * self.y_train.shape[0]
                sample_weights[jet_indices] = np.sqrt(class_weight)

        sample_weights /= np.mean(sample_weights)

        if self.X_train["event_weight"] is not None:
            sample_weights *= self.X_train["event_weight"] / np.mean(
                self.X_train["event_weight"]
            )

        if self.sample_weights is not None:
            self.sample_weights *= sample_weights
        else:
            self.sample_weights = sample_weights

    def enhance_region(
        self, variable, data_type, low_cut=None, high_cut=None, factor=1.0
    ):
        """
        Adjusts sample weights for a specific feature within a given data type based on specified cut-off values.
        Parameters:
            variable (str): The name of the feature to enhance.
            data_type (str): The type of data containing the feature. Must be one of 'jet', 'lepton', 'non_training', or 'global'.
            low_cut (float, optional): The lower bound for the feature values. Only samples with feature values greater than this will be enhanced. Defaults to None.
            high_cut (float, optional): The upper bound for the feature values. Only samples with feature values less than this will be enhanced. Defaults to None.
            factor (float, optional): The factor by which to multiply the sample weights for the selected region. Defaults to 1.0.
        Raises:
            ValueError: If `data_type` is not found in `feature_index_dict`.
            ValueError: If `variable` is not found in the features of the specified `data_type`.
            ValueError: If neither `low_cut` nor `high_cut` is provided.
        Notes:
            - If both `low_cut` and `high_cut` are provided, the weights are adjusted for samples where the feature value is within the range (`low_cut`, `high_cut`).
            - If only `low_cut` is provided, the weights are adjusted for samples where the feature value is greater than `low_cut`.
            - If only `high_cut` is provided, the weights are adjusted for samples where the feature value is less than `high_cut`.
        """

        if self.sample_weights is None:
            self.sample_weights = np.ones(self.y_train.shape[0])
        if data_type not in self.feature_index_dict:
            raise ValueError(
                f"Data type {data_type} not found. Use 'jet', 'lepton', 'non_training', or 'global'."
            )
        if variable not in self.feature_index_dict[data_type]:
            raise ValueError(f"Feature {variable} not found in {data_type} features.")
        if low_cut is None and high_cut is None:
            raise ValueError("At least one of low_cut or high_cut must be provided.")
        if low_cut is not None and high_cut is not None:
            self.sample_weights[
                self.X_train[data_type][:, self.feature_index_dict[data_type][variable]]
                < high_cut
                & self.X_train[data_type][
                    :, self.feature_index_dict[data_type][variable]
                ]
                > low_cut
            ] *= factor
        elif low_cut is not None:
            self.sample_weights[
                self.X_train[data_type][:, self.feature_index_dict[data_type][variable]]
                > low_cut
            ] *= factor
        elif high_cut is not None:
            self.sample_weights[
                self.X_train[data_type][:, self.feature_index_dict[data_type][variable]]
                < high_cut
            ] *= factor

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

        print(f"Model saved to {file_path}.")

    def predict_indices(self, data, exclusive=True):
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
        batch_size = predictions.shape[0]
        one_hot = np.zeros((batch_size, self.max_jets, 2), dtype=int)
        if exclusive:
            for i in range(batch_size):
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
                np.arange(batch_size), np.argmax(predictions[:, :, 0], axis=1), 0
            ] = 1
            one_hot[
                np.arange(batch_size), np.argmax(predictions[:, :, 1], axis=1), 1
            ] = 1
        return one_hot

    def duplicate_jets(self):
        """
        Calculates the fraction of events where the predicted jet indices for two
        different predictions are the same.
        This method uses the trained model to make predictions on the test dataset
        and compares the predicted indices of jets for two different predictions
        to determine how often they match.
        Returns:
            float: The mean fraction of events where the predicted jet indices
            for two different predictions are identical.
        Raises:
            ValueError: If the model has not been built using the `build_model()` method.
            ValueError: If the test data (`X_test` or `y_test`) has not been loaded.
        """

        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )
        if self.X_test is None or self.y_test is None:
            raise ValueError("Data not loaded. Please prepare data using method.")

        if self.global_features is not None:
            y_pred = self.model.predict(
                [self.X_test["jet"], self.X_test["lepton"], self.X_test["global"]],
                verbose=0,
            )
        else:
            y_pred = self.model.predict(
                [self.X_test["jet"], self.X_test["lepton"]], verbose=0
            )

        y_pred = self.predict_indices(self.X_test, exclusive=False)
        return np.mean(
            np.argmax(y_pred[:, :, 0], axis=1) == np.argmax(y_pred[:, :, 1], axis=1),
            axis=0,
        )

    def plot_history(self):
        """
        Plots the training and validation history of the model.
        This method visualizes the training and validation metrics over epochs
        for keys in the training history that have corresponding validation keys.
        For example, if the training history contains a key "accuracy" and a
        corresponding "val_accuracy", it will plot both on the same subplot.
        Returns:
            tuple: A tuple containing the matplotlib figure and axes objects
                   used for the plots.
        Notes:
            - If no training history is available (`self.history` is None),
              the method prints a message and returns without plotting.
            - The method dynamically creates subplots for each metric that has
              both training and validation data.
        """

        if self.history is None:
            print("No training history available.")
            return
        keys = self.history.history.keys()
        accept_keys = []
        for key in keys:
            if "val" not in key and "val_" + key in keys:
                accept_keys.append(key)
        fig, ax = plt.subplots(len(accept_keys), 1, figsize=(5, 5 * len(accept_keys)))
        for i, key in enumerate(accept_keys):
            ax[i].plot(self.history.history[key], label="train")
            ax[i].plot(self.history.history["val_" + key], label="validation")
            ax[i].set_title(key)
            ax[i].set_xlabel("Epochs")
            ax[i].set_ylabel(key)
            ax[i].legend()
        return fig, ax

    def compute_permutation_importance(self, shuffle_number=1):
        def compute_permutation_importance(self, shuffle_number=1):
            """
            Computes the permutation importance of features for the model.
            This method evaluates the importance of each feature by shuffling its values
            and measuring the impact on the model's accuracy. The importance is calculated
            for three accuracy metrics: lepton 1, lepton 2, and combined accuracy.
            Args:
                shuffle_number (int, optional): The number of times to shuffle each feature
                    for calculating the importance. Defaults to 1.
            Returns:
                tuple: A tuple containing three pandas DataFrames:
                    - importance_scores_lep_1: Importance scores for lepton 1 accuracy.
                    - importance_scores_lep_2: Importance scores for lepton 2 accuracy.
                    - importance_scores_combined: Importance scores for combined accuracy.
                    Each DataFrame contains the mean and standard deviation of the importance
                    scores for each feature.
            Raises:
                ValueError: If the model is not built, or the test labels or feature data
                    are not prepared.
                ValueError: If a feature is not found in the jet, lepton, or global feature
                    dictionaries.
            Notes:
                - The method assumes that the model, test data, and feature dictionaries
                  are already prepared.
                - The method uses the `evaluate_accuracy` function to compute the accuracy
                  metrics before and after shuffling the features.
                - The importance scores are calculated as the difference between the base
                  accuracy and the accuracy after shuffling, averaged over the number of
                  shuffles.
            """

        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )
        if self.y_test is None:
            raise ValueError(
                "Labels not prepared. Please prepare data using prepare_data() method."
            )
        if self.X_test is None:
            raise ValueError(
                "Feature data not prepared. Please prepare data using prepare_data() method."
            )
        features = list(self.feature_index_dict["jet"].keys()) + list(
            self.feature_index_dict["lepton"].keys()
        )
        if self.global_features is not None:
            features += list(self.feature_index_dict["global"].keys())

        importance_scores_lep_1 = pd.DataFrame(index=features, columns=["mean", "std"])
        importance_scores_lep_2 = pd.DataFrame(index=features, columns=["mean", "std"])
        importance_scores_combined = pd.DataFrame(
            index=features, columns=["mean", "std"]
        )
        ground_truth = self.y_test.copy()
        lep_1_base_accuracy, lep_2_base_accuracy, combined_base_accuracy = (
            self.evaluate_accuracy()
        )

        lep_1_accuracy = np.zeros(shuffle_number)
        lep_2_accuracy = np.zeros(shuffle_number)
        combined_accuracy = np.zeros(shuffle_number)

        for feature in features:
            X_test_permuted = {}
            for data in self.X_test:
                X_test_permuted[data] = self.X_test[data].copy()
            for shuffle_index in range(shuffle_number):
                if feature in self.feature_index_dict["jet"]:
                    feature_index = self.feature_index_dict["jet"][feature]
                    mask = (
                        X_test_permuted["jet"][:, :, feature_index]
                        != self.padding_value
                    )
                    permuted_values = np.random.permutation(
                        X_test_permuted["jet"][:, :, feature_index][mask]
                    )
                    X_test_permuted["jet"][:, :, feature_index][mask] = permuted_values
                elif feature in self.feature_index_dict["lepton"]:
                    feature_index = self.feature_index_dict["lepton"][feature]
                    X_test_permuted["lepton"][:, :, feature_index] = (
                        np.random.permutation(
                            X_test_permuted["lepton"][:, :, feature_index]
                        )
                    )
                elif feature in self.feature_index_dict["global"]:
                    feature_index = self.feature_index_dict["global"][feature]
                    X_test_permuted["global"][:, :, feature_index] = (
                        np.random.permutation(
                            X_test_permuted["global"][:, :, feature_index]
                        )
                    )
                else:
                    raise ValueError(
                        f"Feature {feature} not found in jet, lepton, or global features."
                    )
                (
                    lep_1_accuracy[shuffle_index],
                    lep_2_accuracy[shuffle_index],
                    combined_accuracy[shuffle_index],
                ) = self.evaluate_accuracy(X_test_permuted, self.y_test)
            print(f"Feature: {feature} computation done.")
            importance_scores_lep_1.loc[feature, "mean"] = np.mean(
                lep_1_base_accuracy - lep_1_accuracy
            )
            importance_scores_lep_1.loc[feature, "std"] = np.std(
                lep_1_base_accuracy - (lep_1_accuracy)
            ) / np.sqrt(shuffle_number)
            importance_scores_lep_2.loc[feature, "mean"] = np.mean(
                lep_2_base_accuracy - (lep_2_accuracy)
            )
            importance_scores_lep_2.loc[feature, "std"] = np.std(
                lep_2_base_accuracy - (lep_2_accuracy)
            ) / np.sqrt(shuffle_number)
            importance_scores_combined.loc[feature, "mean"] = np.mean(
                combined_base_accuracy - combined_accuracy
            )
            importance_scores_combined.loc[feature, "std"] = np.std(
                combined_base_accuracy - (combined_accuracy)
            ) / np.sqrt(shuffle_number)
        return (
            importance_scores_lep_1,
            importance_scores_lep_2,
            importance_scores_combined,
        )

    def plot_permutation_importance(self, shuffle_number, file_name=None):
        """
        Plots the permutation importance scores for three categories: Lepton 1, Lepton 2, and Combined.
        This method computes the permutation importance scores using the `compute_permutation_importance` method,
        sorts the scores in descending order by their mean values, and generates a bar plot for each category
        with error bars representing the standard deviation.
        Args:
            shuffle_number (int): The number of shuffles to perform for computing permutation importance.
            file_name (str, optional): The file path to save the generated plot. If None, the plot is not saved.
        Returns:
            tuple: A tuple containing:
                - fig (matplotlib.figure.Figure): The figure object of the generated plot.
                - ax (numpy.ndarray): An array of axes objects corresponding to the subplots.
        Notes:
            - The plot consists of three subplots:
                1. Lepton 1 Accuracy
                2. Lepton 2 Accuracy
                3. Combined Accuracy
            - Each bar represents the mean importance score, and the error bars represent the standard deviation.
            - If `file_name` is provided, the plot is saved to the specified file path.
        """

        importance_scores_lep_1, importance_scores_lep_2, importance_scores_combined = (
            self.compute_permutation_importance(shuffle_number)
        )
        importance_scores_lep_1 = importance_scores_lep_1.sort_values(
            ascending=False, by="mean"
        )
        importance_scores_lep_2 = importance_scores_lep_2.sort_values(
            ascending=False, by="mean"
        )
        importance_scores_combined = importance_scores_combined.sort_values(
            ascending=False, by="mean"
        )
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        importance_scores_lep_1["mean"].plot(
            kind="bar",
            yerr=importance_scores_lep_1["std"],
            ax=ax[0],
            color="blue",
            alpha=0.7,
        )
        importance_scores_lep_2["mean"].plot(
            kind="bar",
            yerr=importance_scores_lep_2["std"],
            ax=ax[1],
            color="orange",
            alpha=0.7,
        )
        importance_scores_combined["mean"].plot(
            kind="bar",
            yerr=importance_scores_combined["std"],
            ax=ax[2],
            color="green",
            alpha=0.7,
        )
        ax[0].set_title("Lepton 1 Accuracy")
        ax[1].set_title("Lepton 2 Accuracy")
        ax[2].set_title("Combined Accuracy")
        ax[0].set_ylabel("Importance Score")
        ax[1].set_ylabel("Importance Score")
        ax[2].set_ylabel("Importance Score")
        fig.tight_layout()
        if file_name is not None:
            fig.savefig(file_name)
            print(f"Permutation importance plotted and saved to {file_name}.")
        return fig, ax

    def accuracy_vs_feature(
        self, feature_name, data_type="non_training", bins=50, xlims=None
    ):
        def accuracy_vs_feature(
            self, feature_name, data_type="non_training", bins=50, xlims=None
        ):
            """
            Plots the accuracy of the model as a function of a specific feature.
            Parameters:
            -----------
            feature_name : str
                The name of the feature for which the accuracy is to be plotted.
            data_type : str, optional
                The type of data to use for the feature. Must be one of 'jet', 'lepton',
                'non_training', or 'global'. Default is "non_training".
            bins : int, optional
                The number of bins to use for the histogram. Default is 50.
            xlims : tuple, optional
                The x-axis limits for the plot in the form (xmin, xmax). Default is None.
            Raises:
            -------
            ValueError
                If the specified data_type is not found in the feature index dictionary.
            ValueError
                If the specified feature_name is not found in the given data_type.
            ValueError
                If the model has not been built.
            Returns:
            --------
            matplotlib.figure.Figure
                A plot showing the accuracy of the model as a function of the specified feature.
            """

        if data_type not in self.feature_index_dict:
            raise ValueError(
                f"Data type {data_type} not found. Use 'jet', 'lepton', 'non_training', or 'global'."
            )

        if feature_name not in self.feature_index_dict[data_type]:
            raise ValueError(
                f"Feature {feature_name} not found in non-training features."
            )

        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )

        if self.global_features is not None:
            y_pred = self.model.predict(
                [
                    self.X_test["jet"],
                    self.X_test["lepton"],
                    self.X_test["global"],
                ],
                verbose=0,
            )
        else:
            y_pred = self.model.predict(
                [self.X_test["jet"], self.X_test["lepton"]], verbose=0
            )
        y_pred = self.predict_indices(self.X_test)
        feature_data = self.X_test[data_type][
            :, self.feature_index_dict[data_type][feature_name]
        ]
        prediction = np.array(
            [np.argmax(y_pred[:, :, 0], axis=1), np.argmax(y_pred[:, :, 1], axis=1)],
            dtype=int,
        ).T

        return self.plot_accuracy_feature(
            np.argmax(self.y_test, axis=1),
            prediction,
            feature_data,
            feature_name,
            bins=bins,
            xlims=xlims,
            label="RNN Matcher",
            event_weights=self.X_test["event_weight"],
        )

    def evaluate_accuracy(self, data=None, predictions=None):
        """
        Evaluate the accuracy of predictions against the provided data.
        This method computes the accuracy by comparing the predicted indices
        with the actual indices from the provided data. If no data or predictions
        are provided, it defaults to using the test data (`self.X_test`) and
        test labels (`self.y_test`).
        Args:
            data (numpy.ndarray, optional): The input data for evaluation.
                Defaults to None.
            predictions (numpy.ndarray, optional): The predicted values for
                the input data. Defaults to None.
        Returns:
            float: The computed accuracy as a float value.
        Raises:
            ValueError: If only one of `data` or `predictions` is provided.
            ValueError: If neither `data` nor `predictions` is provided and
                the test data (`self.X_test` or `self.y_test`) is not loaded.
        """

        if (data is None) ^ (predictions is None):
            raise ValueError(
                "Either both data and predictions must be provided, or neither."
            )
        if data is None and predictions is None:
            if self.X_test is None or self.y_test is None:
                raise ValueError("Data not loaded. Please prepare data using method.")
            data = self.X_test
            predictions = self.y_test

        return self.compute_accuracy(
            np.argmax(predictions, axis=1),
            np.argmax(self.predict_indices(data), axis=1),
        )

    @staticmethod
    def compute_accuracy(truth, prediction):
        """
        Computes the accuracy of predictions for two lepton categories and their combined accuracy.
        Parameters:
            truth (numpy.ndarray): A 2D array where each row contains the ground truth labels
                                   for two leptons (lep_1 and lep_2).
            prediction (numpy.ndarray): A 2D array where each row contains the predicted labels
                                        for two leptons (lep_1 and lep_2).
        Returns:
            tuple: A tuple containing:
                - lep_1_accuracy (float): The accuracy of predictions for the first lepton.
                - lep_2_accuracy (float): The accuracy of predictions for the second lepton.
                - combined_accuracy (float): The accuracy of predictions where both leptons
                                             are correctly predicted simultaneously.
        """

        lep_1_truth, lep_2_truth = truth[:, 0], truth[:, 1]
        lep_1_pred, lep_2_pred = prediction[:, 0], prediction[:, 1]

        lep_1_accuracy = np.mean(lep_1_truth == lep_1_pred)
        lep_2_accuracy = np.mean(lep_2_truth == lep_2_pred)
        combined_accuracy = np.mean(
            (lep_1_truth == lep_1_pred) & (lep_2_truth == lep_2_pred)
        )

        return lep_1_accuracy, lep_2_accuracy, combined_accuracy

    def get_binned_accuracy(
        self, feature_name, data_type="non_training", bins=20, xlims=None
    ):
        """
        Calculate the binned accuracy of the model predictions for a specific feature.
        This method computes the accuracy of the model predictions within specified bins
        of a given feature. It supports different data types and handles both global and
        non-global features.
        Args:
            feature_name (str): The name of the feature for which the binned accuracy
                is to be calculated.
            data_type (str, optional): The type of data to use. Must be one of
                'jet', 'lepton', 'non_training', or 'global'. Defaults to "non_training".
            bins (int, optional): The number of bins to divide the feature range into.
                Defaults to 20.
            xlims (tuple, optional): A tuple specifying the lower and upper limits of
                the feature range for binning. If None, the range is determined
                automatically. Defaults to None.
        Raises:
            ValueError: If the specified data_type is not found in the feature index dictionary.
            ValueError: If the specified feature_name is not found in the selected data_type.
            ValueError: If the model has not been built.
        Returns:
            np.ndarray: The binned accuracy values for the specified feature.
        """

        if data_type not in self.feature_index_dict:
            raise ValueError(
                f"Data type {data_type} not found. Use 'jet', 'lepton', 'non_training', or 'global'."
            )

        if feature_name not in self.feature_index_dict[data_type]:
            raise ValueError(
                f"Feature {feature_name} not found in non-training features."
            )

        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )

        if self.global_features is not None:
            y_pred = self.model.predict(
                [
                    self.X_test["jet"],
                    self.X_test["lepton"],
                    self.X_test["global"],
                ],
                verbose=0,
            )
        else:
            y_pred = self.model.predict(
                [self.X_test["jet"], self.X_test["lepton"]], verbose=0
            )
        y_pred = self.predict_indices(self.X_test)
        feature_data = self.X_test[data_type][
            :, self.feature_index_dict[data_type][feature_name]
        ]
        prediction = np.array(
            [np.argmax(y_pred[:, :, 0], axis=1), np.argmax(y_pred[:, :, 1], axis=1)],
            dtype=int,
        ).T

        return AssignmentBaseModel.compute_binned_accuracy(
            np.argmax(self.y_test, axis=1),
            prediction,
            feature_data,
            bins=bins,
            xlims=xlims,
            event_weights=self.X_test["event_weight"],
        )

    @staticmethod
    def compute_binned_accuracy(
        truth: np.ndarray,
        prediction: np.ndarray,
        feature_data: np.ndarray,
        bins=20,
        xlims=None,
        event_weights=None,
    ):
        def compute_binned_accuracy(
            truth: np.ndarray,
            prediction: np.ndarray,
            feature_data: np.ndarray,
            bins=20,
            xlims=None,
            event_weights=None,
        ):
            """
            Compute the binned accuracy of predictions compared to truth values, based on a feature's distribution.
            Parameters:
            ----------
            truth : np.ndarray
                A 2D array where each row contains the true labels for two lepton predictions.
            prediction : np.ndarray
                A 2D array where each row contains the predicted labels for two leptons.
            feature_data : np.ndarray
                A 1D array of feature values used for binning.
            bins : int, optional
                The number of bins to divide the feature data into. Default is 20.
            xlims : tuple, optional
                A tuple specifying the range (min, max) of the feature data to consider for binning.
                If None, the range is determined automatically. Default is None.
            event_weights : np.ndarray, optional
                A 1D array of weights for each event. If None, all events are weighted equally. Default is None.
            Returns:
            -------
            binned_accuracy_combined : np.ndarray
                A 1D array containing the accuracy for each bin.
            feature_bins : np.ndarray
                The edges of the bins used for the feature data.
            feature_hist : np.ndarray
                The histogram of the feature data, normalized to form a probability density.
            Notes:
            -----
            - The function calculates the accuracy for each bin by comparing the predicted labels
              to the true labels for both leptons.
            - The binning is performed based on the `feature_data` array, and the accuracy is
              weighted by `event_weights` if provided.
            - A small epsilon (1e-9) is added to the denominator to avoid division by zero.
            """

        lep_1_truth, lep_2_truth = truth[:, 0], truth[:, 1]
        lep_1_pred, lep_2_pred = prediction[:, 0], prediction[:, 1]
        if xlims is None:
            feature_hist, feature_bins = np.histogram(
                feature_data, bins=bins, density=True, weights=event_weights
            )
        else:
            feature_hist, feature_bins = np.histogram(
                feature_data,
                bins=bins,
                density=True,
                weights=event_weights,
                range=xlims,
            )

        xlims = (feature_bins[0], feature_bins[-1])

        binning_mask = (
            (feature_data.reshape(-1, 1) <= feature_bins[1:].reshape(1, -1))
            & (feature_data.reshape(-1, 1) >= feature_bins[:-1].reshape(1, -1))
        ).astype(float)
        binned_accuracy_combined = np.sum(
            np.broadcast_to(
                ((lep_1_pred == lep_1_truth) & (lep_2_pred == lep_2_truth)).reshape(
                    -1, 1
                ),
                (lep_2_pred.shape[0], bins),
            )
            * binning_mask,
            axis=0,
        ) / (np.sum(binning_mask, axis=0) + 1e-9)

        return binned_accuracy_combined, feature_bins, feature_hist

    @staticmethod
    def plot_accuracy_feature(
        truth,
        prediction,
        feature_data,
        feature_name,
        bins=20,
        xlims=None,
        fig=None,
        ax=None,
        accuracy_color="tab:blue",
        label=None,
        event_weights=None,
    ):
        def plot_accuracy_feature(
            truth,
            prediction,
            feature_data,
            feature_name,
            bins=20,
            xlims=None,
            fig=None,
            ax=None,
            accuracy_color="tab:blue",
            label=None,
            event_weights=None,
        ):
            """
            Plots the accuracy of predictions as a function of a given feature, with optional error bars
            computed using bootstrapping. Also overlays a histogram of the feature counts.
            Parameters:
            ----------
            truth : array-like
                Ground truth labels for the data.
            prediction : array-like
                Predicted labels or probabilities for the data.
            feature_data : array-like
                Feature values corresponding to the data points.
            feature_name : str
                Name of the feature to be displayed on the x-axis.
            bins : int, optional
                Number of bins to divide the feature data into (default is 20).
            xlims : tuple, optional
                Limits for the x-axis as (min, max). If None, limits are determined automatically (default is None).
            fig : matplotlib.figure.Figure, optional
                Existing figure object to plot on. If None, a new figure is created (default is None).
            ax : matplotlib.axes.Axes, optional
                Existing axes object to plot on. If None, a new axes is created (default is None).
            accuracy_color : str, optional
                Color for the accuracy plot (default is "tab:blue").
            label : str, optional
                Label for the accuracy plot, used in the legend (default is None).
            event_weights : array-like, optional
                Weights for the events, used for weighted histogram computation (default is None).
            Returns:
            -------
            fig : matplotlib.figure.Figure
                The figure object containing the plot.
            ax : matplotlib.axes.Axes
                The axes object containing the plot.
            Notes:
            -----
            - The function uses bootstrapping to compute error bars for the accuracy values.
            - A secondary y-axis is used to display the histogram of feature counts.
            - The function relies on `AssignmentBaseModel.compute_binned_accuracy` for accuracy computation.
            """

        # Use compute_binned_accuracy to get binned accuracies
        binned_accuracy_combined, feature_bins, feature_hist = (
            AssignmentBaseModel.compute_binned_accuracy(
                truth,
                prediction,
                feature_data,
                bins=bins,
                xlims=xlims,
                event_weights=None,
            )
        )
        # Compute histogram for feature counts (optionally weighted)
        centers = (feature_bins[:-1] + feature_bins[1:]) / 2
        # Bootstrapping for error bars
        n_bootstrap = 10
        rng = np.random.default_rng()
        acc_comb_boot = np.zeros((n_bootstrap, bins))
        for i in range(n_bootstrap):
            indices = rng.integers(0, len(feature_data), len(feature_data))
            truth_bs = truth[indices]
            pred_bs = prediction[indices]
            feat_bs = feature_data[indices]
            acc_comb_boot[i], _, _ = AssignmentBaseModel.compute_binned_accuracy(
                truth_bs,
                pred_bs,
                feat_bs,
                bins=bins,
                xlims=(feature_bins[0], feature_bins[-1]),
            )
        err_comb = np.std(acc_comb_boot, axis=0) / np.sqrt(n_bootstrap)

        # Prepare data for plotting

        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            # Plot histogram of feature counts on secondary y-axis
            ax_clone = ax.twinx()
            ax_clone.set_ylabel("Feature Count", color="tab:orange")
            ax_clone.bar(
                centers,
                feature_hist,
                width=np.diff(feature_bins),
                alpha=0.3,
                color="tab:orange",
            )
            ax_clone.tick_params(axis="y", labelcolor="tab:orange")
            ax.set_xlabel(feature_name)
            ax.set_ylim(0, 1.1)
            ax.set_ylabel("Accuracy")
            ax.tick_params(axis="y")
            ax.set_xlim(xlims)
        # Plot accuracy with error bars
        ax.errorbar(
            centers,
            binned_accuracy_combined,
            yerr=err_comb,
            label=label,
            fmt="o",
            capsize=2,
            color=accuracy_color,
        )

        fig.tight_layout()
        return fig, ax

    @staticmethod
    def plot_external_confusion_matrix(
        jet_matcher_indices=None,
        truth_index=None,
        max_jets=None,
        n_bootstrap=100,
        sample_weight=None,
    ):
        def plot_external_confusion_matrix(
            jet_matcher_indices=None,
            truth_index=None,
            max_jets=None,
            n_bootstrap=100,
            sample_weight=None,
        ):
            """
            Plots confusion matrices for two leptons using bootstrap resampling.
            This function computes confusion matrices for two leptons (lepton 1 and lepton 2)
            based on the provided truth and predicted indices. It uses bootstrap resampling
            to calculate the mean and standard deviation of the confusion matrices and
            visualizes them using heatmaps.
            Parameters:
            -----------
            jet_matcher_indices : np.ndarray
                Array of predicted indices for jets, with shape (n_samples, 2), where the
                second dimension corresponds to the two leptons.
            truth_index : np.ndarray
                Array of true indices for jets, with shape (n_samples, 2), where the
                second dimension corresponds to the two leptons.
            max_jets : int
                The maximum number of jet categories (used to define the size of the confusion matrix).
            n_bootstrap : int, optional, default=100
                The number of bootstrap resampling iterations to perform.
            sample_weight : np.ndarray, optional, default=None
                Array of sample weights for each data point. If None, all samples are
                treated equally.
            Returns:
            --------
            fig : matplotlib.figure.Figure
                The figure object containing the confusion matrix plots.
            ax : np.ndarray of matplotlib.axes._subplots.AxesSubplot
                Array of axes objects corresponding to the two confusion matrix plots.
            Notes:
            ------
            - The function normalizes the confusion matrices row-wise (true label axis).
            - The standard deviation of the confusion matrices is displayed as red text
              on the heatmaps.
            - The function uses seaborn for heatmap visualization.
            """

        bootstrap_confusion_lep_1 = np.zeros((n_bootstrap, max_jets, max_jets))
        bootstrap_confusion_lep_2 = np.zeros((n_bootstrap, max_jets, max_jets))

        for i in range(n_bootstrap):
            indices = np.random.choice(len(truth_index), len(truth_index), replace=True)
            truth_resampled = truth_index[indices]
            matcher_resampled = jet_matcher_indices[indices]
            if sample_weight is not None:
                sample_weight_resampled = sample_weight[indices]
            else:
                sample_weight_resampled = None

            confusion_lep_1_resampled = confusion_matrix(
                truth_resampled[:, 0],
                matcher_resampled[:, 0],
                labels=np.arange(max_jets),
                sample_weight=sample_weight_resampled,
            )
            confusion_lep_2_resampled = confusion_matrix(
                truth_resampled[:, 1],
                matcher_resampled[:, 1],
                labels=np.arange(max_jets),
                sample_weight=sample_weight_resampled,
            )

            bootstrap_confusion_lep_1[i] = (
                confusion_lep_1_resampled.astype("float")
                / confusion_lep_1_resampled.sum(axis=1)[:, np.newaxis]
            )
            bootstrap_confusion_lep_2[i] = (
                confusion_lep_2_resampled.astype("float")
                / confusion_lep_2_resampled.sum(axis=1)[:, np.newaxis]
            )

        mean_confusion_lep_1 = bootstrap_confusion_lep_1.mean(axis=0)
        mean_confusion_lep_2 = bootstrap_confusion_lep_2.mean(axis=0)
        std_confusion_lep_1 = bootstrap_confusion_lep_1.std(axis=0)
        std_confusion_lep_2 = bootstrap_confusion_lep_2.std(axis=0)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        baseline_confusion_lep_1 = (
            confusion_matrix(
                truth_index[:, 0],
                jet_matcher_indices[:, 0],
                labels=np.arange(max_jets),
                sample_weight=sample_weight_resampled,
            ).astype("float")
            / confusion_matrix(
                truth_index[:, 0],
                jet_matcher_indices[:, 0],
                labels=np.arange(max_jets),
                sample_weight=sample_weight_resampled,
            ).sum(axis=1)[:, np.newaxis]
        )
        baseline_confusion_lep_2 = (
            confusion_matrix(
                truth_index[:, 1],
                jet_matcher_indices[:, 1],
                labels=np.arange(max_jets),
                sample_weight=sample_weight_resampled,
            ).astype("float")
            / confusion_matrix(
                truth_index[:, 1],
                jet_matcher_indices[:, 1],
                labels=np.arange(max_jets),
                sample_weight=sample_weight_resampled,
            ).sum(axis=1)[:, np.newaxis]
        )
        sns.heatmap(
            baseline_confusion_lep_1,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar=False,
            ax=ax[0],
        )
        sns.heatmap(
            baseline_confusion_lep_2,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar=False,
            ax=ax[1],
        )

        for i in range(max_jets):
            for j in range(max_jets):
                ax[0].text(
                    j + 0.5 + 0.3,
                    i + 0.5,
                    f"±{std_confusion_lep_1[i, j]:.2f}",
                    color="red",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
                ax[1].text(
                    j + 0.5 + 0.3,
                    i + 0.5,
                    f"±{std_confusion_lep_2[i, j]:.2f}",
                    color="red",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        ax[0].set_title("Confusion Matrix for Lepton 1 (Bootstrap)")
        ax[1].set_title("Confusion Matrix for Lepton 2 (Bootstrap)")
        ax[0].set_xlabel("Predicted Label")
        ax[1].set_xlabel("Predicted Label")
        ax[0].set_ylabel("True Label")
        ax[1].set_ylabel("True Label")
        return fig, ax

    def plot_confusion_matrix(self, n_bootstrap=100, exclusive=True):
        def plot_confusion_matrix(self, n_bootstrap=100, exclusive=True):
            """
            Plots the confusion matrix for the model's predictions on the test dataset.
            This method computes the confusion matrix by comparing the predicted indices
            with the ground truth indices from the test dataset. It supports bootstrapping
            for uncertainty estimation and allows for exclusive matching of predictions.
            Args:
                n_bootstrap (int, optional): The number of bootstrap samples to use for
                    uncertainty estimation. Defaults to 100.
                exclusive (bool, optional): If True, uses exclusive matching for predictions.
                    Defaults to True.
            Returns:
                matplotlib.figure.Figure: A figure object containing the plotted confusion matrix.
            Notes:
                - The method uses `predict_indices` to compute the predicted indices.
                - The ground truth indices are derived from `y_test`.
                - The `max_jets` attribute is used to determine the maximum number of jets.
                - The `plot_external_confusion_matrix` method is called to generate the plot.
                - The `event_weight` column in `X_test` is used as sample weights for the plot.
            """

        jet_matcher_indices = np.argmax(
            self.predict_indices(self.X_test, exclusive=exclusive), axis=1
        )
        truth_index = np.argmax(self.y_test, axis=1)
        max_jets = self.max_jets
        return self.plot_external_confusion_matrix(
            jet_matcher_indices,
            truth_index,
            max_jets,
            n_bootstrap=n_bootstrap,
            sample_weight=self.X_test["event_weight"],
        )

    def get_confusion_matrix(
        self, data=None, labels=None, sample_weights=None, exclusive=True
    ):
        """
        Computes and returns the normalized confusion matrices for two leptons based on the predicted and true labels.
        Parameters:
        -----------
        data : array-like, optional
            The input data for prediction. If None, `self.X_test` is used.
        labels : array-like, optional
            The true labels corresponding to the input data. If None, `self.y_test` is used.
        sample_weights : array-like, optional
            Weights for each sample. If None, and "event_weight" exists in `self.X_test`, it is used.
        exclusive : bool, default=True
            If True, ensures exclusive predictions for each class.
        Returns:
        --------
        normed_lep_1_confusion_matrix : numpy.ndarray
            The normalized confusion matrix for the first lepton.
        normed_lep_2_confusion_matrix : numpy.ndarray
            The normalized confusion matrix for the second lepton.
        Raises:
        -------
        ValueError
            If only one of `data` or `labels` is provided while the other is None.
        Notes:
        ------
        - The confusion matrices are normalized such that each row sums to 1.
        - The method assumes that the labels are one-hot encoded and computes the
          indices of the maximum values along the last axis for both predictions
          and true labels.
        """

        if data is None and labels is None and sample_weights is None:
            data = self.X_test
            labels = self.y_test
            if "event_weight" in self.X_test:
                sample_weights = self.X_test["event_weight"]
        elif data is None or labels is None:
            raise ValueError(
                "Both data and labels must be provided or both must be None."
            )

        jet_matcher_indices = np.argmax(
            self.predict_indices(data, exclusive=exclusive), axis=1
        )
        truth_index = np.argmax(labels, axis=1)
        max_jets = self.max_jets
        lep_1_confusion_matrix = confusion_matrix(
            truth_index[:, 0],
            jet_matcher_indices[:, 0],
            labels=np.arange(max_jets),
            sample_weight=sample_weights,
        ).astype("float")
        lep_2_confusion_matrix = confusion_matrix(
            truth_index[:, 1],
            jet_matcher_indices[:, 1],
            labels=np.arange(max_jets),
            sample_weight=sample_weights,
        ).astype("float")
        normed_lep_1_confusion_matrix = (
            lep_1_confusion_matrix / lep_1_confusion_matrix.sum(axis=1)[:, np.newaxis]
        )
        normed_lep_2_confusion_matrix = (
            lep_2_confusion_matrix / lep_2_confusion_matrix.sum(axis=1)[:, np.newaxis]
        )

        return normed_lep_1_confusion_matrix, normed_lep_2_confusion_matrix
