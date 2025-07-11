import tensorflow as tf
import keras
import sklearn as sk
import numpy as np
from .DataLoader import DataPreprocessor
from . import CustomObjects

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow import convert_to_tensor
from copy import deepcopy


class RegressionBaseModel:
    """
    RegressionBaseModel is a base class for building and training regression models with support for custom data preprocessing, 
    model saving/loading, and evaluation.
    Attributes:
        model (keras.Model): The Keras model instance.
        X_train (np.ndarray): Training input data.
        X_test (np.ndarray): Testing input data.
        y_train (np.ndarray): Training target data.
        y_test (np.ndarray): Testing target data.
        history (keras.callbacks.History): Training history object.
        sample_weights (np.ndarray): Sample weights for training.
        class_weights (np.ndarray): Class weights for training.
        max_leptons (int): Maximum number of leptons in the data.
        max_jets (int): Maximum number of jets in the data.
        global_features (list): List of global features in the data.
        n_jets (int): Number of jet features.
        n_leptons (int): Number of lepton features.
        n_global (int): Number of global features.
        n_regression_targets (int): Number of regression targets.
        padding_value (float): Padding value used in the data.
        combined_features (list): List of combined features.
        n_combined (int): Number of combined features.
        feature_index_dict (dict): Dictionary mapping feature names to indices.
        data_preprocessor (DataPreprocessor): Data preprocessor instance.
    Methods:
        __init__(data_preprocessor):
            Initializes the RegressionBaseModel with a data preprocessor.
        load_data(X_train=None, y_train=None, X_test=None, y_test=None):
            Loads the training and testing data into the model.
        save_model(file_path="model.keras"):
            Saves the model to the specified file path.
        load_model(file_path):
            Loads a model from the specified file path.
        build_model():
            Abstract method to build the model architecture. Must be implemented by subclasses.
        plot_history():
            Plots the training history of the model.
        compile_model(lambda_excl=1, *args, **kwargs):
            Compiles the model with specified loss functions and additional arguments.
        train_model(epochs=10, batch_size=32, weight=None, *args, **kwargs):
            Trains the model with the provided data and parameters.
        compute_sample_weights(alpha=0.5):
            Computes sample weights for training based on data distribution.
        scatter_plot(bins=30, fig=None, ax=None):
            Generates a scatter plot comparing predicted and true regression values.
        confusion_matrix():
            Generates a confusion matrix for assignment predictions.
        make_prediction(data, exclusive=True):
            Makes predictions using the model for the given data.
        evaluate_accuracy(data=None, truth=None, exclusive=True):
            Evaluates the accuracy of the model's predictions.
        compute_assignment_accuracy(truth, prediction):
            Computes the accuracy of assignment predictions.
        compute_regression_accuracy(truth, prediction):
            Computes the regression accuracy as the mean absolute error.
        plot_confusion_matrix(n_bootstrap=100, exclusive=True):
            Plots a confusion matrix with bootstrap confidence intervals.
        get_confusion_matrix(exclusive=True):
            Computes confusion matrices for lepton assignments.
        plot_external_confusion_matrix(jet_matcher_indices, truth_index, max_jets, n_bootstrap=100, sample_weight=None):
            Plots an external confusion matrix with bootstrap confidence intervals.
        compute_binned_regression_accuracy(truth, prediction, feature_data, bins=20, xlims=None, event_weights=None):
            Computes binned regression accuracy for a given feature.
        get_binned_regression_accuracy(feature_name, data_type="non_training", bins=20, xlims=None):
            Computes binned accuracy for a specific feature.
        compute_permutation_importance(shuffle_number=1):
            Computes permutation importance for each feature in the model.
        get_outputs():
            Returns the outputs of the model, including predictions and ground truth.
    """
    def __init__(self, data_preprocessor: DataPreprocessor):
        """
        Initializes the RegressionBaseModel with the given data preprocessor.
        Args:
            data_preprocessor (DataPreprocessor): An instance of DataPreprocessor 
                containing the preprocessed data and metadata required for the model.
        Attributes:
            model (keras.Model): The Keras model instance, initialized as None.
            X_train (np.ndarray): Training feature data, initialized as None.
            X_test (np.ndarray): Testing feature data, initialized as None.
            y_train (np.ndarray): Training target data, initialized as None.
            y_test (np.ndarray): Testing target data, initialized as None.
            history (keras.callbacks.History): Training history, initialized as None.
            sample_weights (np.ndarray): Sample weights for training, initialized as None.
            class_weights (np.ndarray): Class weights for training, initialized as None.
            max_leptons (int): Maximum number of leptons from the data preprocessor.
            max_jets (int): Maximum number of jets from the data preprocessor.
            global_features (list): List of global features from the data preprocessor.
            n_jets (int): Number of jet features from the data preprocessor.
            n_leptons (int): Number of lepton features from the data preprocessor.
            n_global (int): Number of global features, or 0 if none are provided.
            n_regression_targets (int): Number of regression targets from the data preprocessor.
            padding_value (float): Padding value used in the data preprocessor.
            combined_features (list): List of combined features from the data preprocessor.
            n_combined (int): Number of combined features from the data preprocessor.
            feature_index_dict (dict): Dictionary mapping feature names to indices.
            data_preprocessor (DataPreprocessor): The data preprocessor instance.
        """

        self.model: keras.Model = None
        self.X_train: np.ndarray = None
        self.X_test: np.ndarray = None
        self.y_train: np.ndarray = None
        self.y_test: np.ndarray = None
        self.history: keras.callbacks.History = None
        self.sample_weights: np.ndarray = None
        self.class_weights: np.ndarray = None
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
        self.n_regression_targets: int = data_preprocessor.n_regression_targets
        self.padding_value: float = data_preprocessor.padding_value
        self.combined_features = data_preprocessor.combined_features
        self.n_combined: int = data_preprocessor.n_combined
        self.feature_index_dict: dict = data_preprocessor.feature_index_dict
        self.data_preprocessor = data_preprocessor

    def load_data(self, X_train=None, y_train=None, X_test=None, y_test=None):
        """
        Loads training and testing data into the model. 
        If all data arguments (X_train, y_train, X_test, y_test) are provided, 
        they will be directly assigned to the corresponding attributes. 
        Otherwise, the data will be fetched using the `data_preprocessor.get_data()` method.
        Args:
            X_train (array-like, optional): Training feature data. Defaults to None.
            y_train (array-like, optional): Training target data. Defaults to None.
            X_test (array-like, optional): Testing feature data. Defaults to None.
            y_test (array-like, optional): Testing target data. Defaults to None.
        Attributes:
            X_train (array-like): Training feature data.
            y_train (array-like): Training target data.
            X_test (array-like): Testing feature data.
            y_test (array-like): Testing target data.
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
        else:
            self.X_train, self.y_train, self.X_test, self.y_test = (
                self.data_preprocessor.get_data()
            )

    def save_model(self, file_path="model.keras"):
        """
        Saves the current model to the specified file path in the `.keras` format and 
        writes the model's structure to a separate text file.
        Args:
            file_path (str): The file path where the model will be saved. 
                             Must end with `.keras`. Defaults to "model.keras".
        Raises:
            ValueError: If the model has not been built using the `build_model()` method.
            ValueError: If the provided file path does not end with `.keras`.
        Side Effects:
            - Saves the model to the specified file path.
            - Writes the model's structure to a text file with the same name as the 
              model file but with `_structure.txt` appended.
        Example:
            >>> model.save_model("my_model.keras")
            This will save the model to "my_model.keras" and write the structure to 
            "my_model_structure.txt".
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
        Load a pre-trained model from the specified file path.
        This method loads a Keras model from the given file path and assigns it 
        to the `self.model` attribute. It uses the `CustomObjects` dictionary 
        to ensure any custom layers, loss functions, or other components are 
        properly loaded.
        Args:
            file_path (str): The path to the file containing the saved Keras model.
        Returns:
            None
        Raises:
            OSError: If the file at `file_path` does not exist or cannot be loaded.
        """

        self.model = keras.models.load_model(
            file_path, custom_objects=CustomObjects.__dict__
        )
        print(f"Model loaded from {file_path}")

    def build_model(self):
        """
        Build the model architecture.
        This method should be implemented by subclasses to define the specific
        architecture and configuration of the model. It is intended to be 
        overridden in derived classes.
        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        
        raise NotImplementedError("Subclasses should implement this method.")

    def plot_history(self):
        """
        Plots the training and validation history of the model.
        This method visualizes the training and validation metrics stored in the 
        `self.history` object. It identifies metrics that have both training and 
        validation values and creates subplots for each of them.
        Returns:
            tuple: A tuple containing the matplotlib figure and axes objects 
                   (fig, ax) used for the plots.
        Notes:
            - The method assumes that `self.history` is an object with a `history` 
              attribute, which is a dictionary containing metric names as keys and 
              lists of values as values.
            - Metrics with "accuracy" in their name are plotted with a linear scale 
              on the y-axis, while others are plotted with a logarithmic scale.
            - If no training history is available (`self.history` is None), the 
              method prints a message and returns without plotting.
        Example:
            >>> model.plot_history()
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
            ax[i].grid()
            ax[i].set_yscale("linear" if "accuracy" in key else "log")
            ax[i].legend()
        return fig, ax

    def compile_model(self, lambda_excl=1, *args, **kwargs):
        """
        Compiles the model with specified loss functions and additional arguments.
        This method sets up the model for training by configuring the loss functions
        for the outputs and passing any additional arguments to the `compile` method
        of the model.
        Args:
            lambda_excl (float, optional): A regularization parameter for the combined loss.
            Defaults to 1.
            *args: Variable length argument list to be passed to the `compile` method.
            **kwargs: Arbitrary keyword arguments to be passed to the `compile` method.
        Notes:
            - The `assignment_output` uses a custom loss function defined in `CustomObjects.AssignmentLoss`.
            - The `regression_output` uses a custom loss function defined in `CustomObjects.RegressionLoss`.
            - The commented-out line indicates an alternative combined loss function
              (`CustomObjects.CombinedLoss`) that can be used with the `lambda_excl` parameter.
        """

        self.model.compile(
            loss={
                "assignment_output": CustomObjects.AssignmentLoss(),
                "regression_output": CustomObjects.RegressionLoss(),
            },
            # loss=CustomObjects.CombinedLoss(lambda_excl=lambda_excl),
            *args,
            **kwargs,
        )

    def train_model(self, epochs=10, batch_size=32, weight=None, *args, **kwargs):
        """
        Trains the model using the provided training data and parameters.
        Args:
            epochs (int, optional): Number of epochs to train the model. Defaults to 10.
            batch_size (int, optional): Number of samples per gradient update. Defaults to 32.
            weight (str, optional): Type of weighting to apply. Can be 'sample' or 'class'. 
                If None, no weighting is applied. Defaults to None.
            *args: Additional positional arguments to pass to the `fit` method of the model.
            **kwargs: Additional keyword arguments to pass to the `fit` method of the model.
        Raises:
            ValueError: If the model is not built or the training data is not loaded.
            ValueError: If an invalid weight type is provided.
        Notes:
            - The method checks if the model and training data are prepared before training.
            - If `weight` is specified, it computes either sample weights or class weights 
              and passes them to the `fit` method.
            - Handles both cases where global features are present or absent in the data.
            - The training history is stored in `self.history`.
        Returns:
            None
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
                {
                    "assignment_output": self.X_test["labels"],
                    "regression_output": self.y_test,
                },
            )
        else:
            inputs = [self.X_train["jet"], self.X_train["lepton"]]
            validation_data = (
                [self.X_test["jet"], self.X_test["lepton"]],
                {
                    "assignment_output": self.X_test["labels"],
                    "regression_output": self.y_test,
                },
            )
        outputs = {
            "assignment_output": self.X_train["labels"],
            "regression_output": self.y_train,
        }

        self.history = self.model.fit(
            inputs,
            outputs,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            *args,
            **kwargs,
        )
        print("Training completed.")

    def compute_sample_weights(self, alpha=0.5):
        """
        Compute sample weights for training based on event weights, jet numbers, 
        and regression target distributions.
        This method calculates two types of sample weights:
        - Assignment sample weights: Adjusted based on the number of jets in each sample 
          and normalized by the mean.
        - Regression sample weights: Adjusted based on the distribution of regression 
          targets using histogram binning and normalized by the mean.
        Args:
            alpha (float, optional): A scaling factor to control the influence of 
                sample frequency on the weights. Default is 0.5.
        Attributes:
            self.sample_weights (dict): A dictionary containing:
                - "assignment_output": The computed assignment sample weights.
                - "regression_output": The computed regression sample weights.
        Notes:
            - The method assumes that `self.X_train` contains the training data with 
              keys "jet" and "event_weight".
            - The `self.y_train` attribute is expected to contain the regression targets.
            - The `self.padding_value` is used to identify padded jet entries.
            - The `self.max_jets` attribute specifies the maximum number of jets.
            - The `self.n_regression_targets` attribute specifies the number of regression 
              targets.
        """

        bins = 5
        jet_numbers = np.sum(
            ~np.all(self.X_train["jet"] == self.padding_value, axis=(-1)), axis=-1
        )

        if self.X_train["event_weight"] is not None:
            event_weights = self.X_train["event_weight"] / np.mean(
                self.X_train["event_weight"]
            )
        else:
            event_weights = np.ones(self.X_train["jet"].shape[0])

        assignment_sample_weights = np.ones(self.X_train["labels"].shape[0])
        regression_sample_weights = np.ones(self.y_train.shape[0])

        assignment_sample_weights *= event_weights
        regression_sample_weights *= event_weights

        for jet_number in range(self.max_jets):
            jet_number_indices = np.where(jet_numbers == jet_number)[0]
            if len(jet_number_indices) == 0:
                continue
            else:
                assignment_sample_weights[jet_number_indices] *= (
                    1.0 / len(jet_number_indices) ** alpha
                )
        assignment_sample_weights /= np.mean(assignment_sample_weights)

        regression_sample_weights = np.ones(self.y_train.shape[0])
        hist, edges = np.histogramdd(self.y_train, bins=bins)  # Avoid division by zero

        hist = hist + 1e-9

        bin_indices = np.ones(
            (self.y_train.shape[0], self.n_regression_targets), dtype=int
        )
        for i in range(self.n_regression_targets):
            bin_indices[:, i] = np.digitize(self.y_train[:, i], edges[i][1:]) - 1
        for i in range(self.y_train.shape[0]):
            bin_index = tuple(bin_indices[i])
            regression_sample_weights[i] = (
                1.0 / hist[bin_index] ** alpha * self.y_train.shape[0]
            )

        regression_sample_weights /= np.mean(regression_sample_weights)
        self.sample_weights = {
            "assignment_output": assignment_sample_weights,
            "regression_output": regression_sample_weights,
        }

    def scatter_plot(self, bins=30, fig=None, ax=None):
        """
        Generates a scatter plot comparing the predicted regression values to the true values.
        This method creates a 2D histogram (heatmap) for each regression target, showing the 
        relationship between the predicted and true values. The histogram is normalized along 
        the true value axis.
        Args:
            bins (int, optional): Number of bins to use for the 2D histogram. Defaults to 30.
            fig (matplotlib.figure.Figure, optional): Existing matplotlib figure to use. 
                If None, a new figure is created. Defaults to None.
            ax (matplotlib.axes.Axes or list of Axes, optional): Existing matplotlib axes to use. 
                If None, new axes are created. Defaults to None.
        Returns:
            tuple: A tuple containing:
                - fig (matplotlib.figure.Figure): The matplotlib figure object.
                - ax (list of matplotlib.axes.Axes): A list of matplotlib axes objects, one for each 
                  regression target.
        Notes:
            - The method assumes that the model has been trained and `self.model.predict` can 
              generate predictions.
            - The `self.X_test` dictionary should contain the keys "jet", "lepton", and "global" 
              for input features, and optionally "event_weight" for weighting the histogram.
            - The `self.y_test` should contain the true regression values.
            - The number of regression targets is determined by `self.n_regression_targets`.
        """

        y_pred = self.model.predict(
            [self.X_test["jet"], self.X_test["lepton"], self.X_test["global"]]
        )
        _, regression_pred = y_pred
        regression_truth = self.y_test
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 6 * self.n_regression_targets))
        if self.n_regression_targets == 1:
            ax = [ax]

        for i in range(self.n_regression_targets):
            range_min = min(
                np.min(regression_truth[:, i]),
                np.min(regression_pred[:, i]),
            )
            range_max = max(
                np.max(regression_truth[:, i]),
                np.max(regression_pred[:, i]),
            )
            hist, xedges, yedges = np.histogram2d(
                regression_truth[:, i],
                regression_pred[:, i],
                bins=bins,
                range=[[range_min, range_max], [range_min, range_max]],
                weights=(
                    self.X_test["event_weight"]
                    if "event_weight" in self.X_test
                    else None
                ),
            )
            hist /= (np.sum(hist, axis=1) + 1e-9)[:, np.newaxis]

            ax[i].imshow(
                hist.T,
                origin="lower",
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                aspect="auto",
                cmap="Blues",
            )
        return fig, ax

    def confusion_matrix(self):
        """
        Generates and visualizes the confusion matrix for the model's predictions.
        This method computes the confusion matrix by comparing the predicted 
        class labels with the true class labels from the test dataset. It then 
        visualizes the confusion matrix as a heatmap using Matplotlib and Seaborn.
        Returns:
            tuple: A tuple containing:
                - fig (matplotlib.figure.Figure): The Matplotlib figure object for the heatmap.
                - ax (matplotlib.axes._subplots.AxesSubplot): The Matplotlib axes object for the heatmap.
        Notes:
            - The method assumes that the model has been trained and that the 
              test dataset (`self.X_test`) contains the necessary keys: "jet", 
              "lepton", "global", and "labels".
            - The predictions are expected to be in the form of a dictionary 
              with a key "assignment_output" containing the predicted probabilities 
              or logits for each class.
        """

        y_pred = self.model.predict(
            [self.X_test["jet"], self.X_test["lepton"], self.X_test["global"]]
        )
        assignment_pred = y_pred["assignment_output"]
        assignment_truth = self.X_test["labels"]

        # Convert predictions to class labels
        assignment_pred_labels = np.argmax(assignment_pred, axis=-1)
        assignment_truth_labels = np.argmax(assignment_truth, axis=-1)

        cm = confusion_matrix(
            assignment_truth_labels.flatten(), assignment_pred_labels.flatten()
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        return fig, ax

    def make_prediction(self, data, exclusive=True):
        """
        Make predictions using the trained model.
        This method takes input data and generates predictions for assignments and 
        regression outputs. It supports both exclusive and non-exclusive assignment 
        modes. Exclusive mode ensures that each jet is assigned to at most one lepton 
        and vice versa.
        Args:
            data (dict): A dictionary containing input data with keys:
                - "jet": Jet features.
                - "lepton": Lepton features.
                - "global" (optional): Global features, if applicable.
            exclusive (bool, optional): If True, ensures exclusive assignments 
                between jets and leptons. Defaults to True.
        Returns:
            tuple: A tuple containing:
                - one_hot (numpy.ndarray): A one-hot encoded array of shape 
                  (batch_size, max_jets, 2) representing the assignment predictions.
                - regression (numpy.ndarray): An array of regression predictions.
        Raises:
            ValueError: If the model is not built before calling this method.
        Warnings:
            Prints a warning if the regression output appears to be constant, 
            indicating potential issues with the model or data.
        """

        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )
        if self.global_features is not None:
            prediction = self.model.predict(
                [data["jet"], data["lepton"], data["global"]], verbose=0
            )
            if isinstance(prediction, dict):
                assignment, regression = (
                    prediction["assignment_output"],
                    prediction["regression_output"],
                )
            elif isinstance(prediction, list):
                assignment, regression = prediction[0], prediction[1]
        else:
            prediction = self.model.predict([data["jet"], data["lepton"]], verbose=0)
            assignment, regression = (
                prediction["assignment_output"],
                prediction["regression_output"],
            )
        batch_size = assignment.shape[0]
        one_hot = np.zeros((batch_size, self.max_jets, 2), dtype=int)
        if exclusive:
            for i in range(batch_size):
                probs = assignment[i].copy()
                for _ in range(self.max_leptons):
                    jet_index, lepton_index = np.unravel_index(
                        np.argmax(probs), probs.shape
                    )
                    one_hot[i, jet_index, lepton_index] = 1
                    probs[jet_index, :] = 0
                    probs[:, lepton_index] = 0
        else:
            one_hot[
                np.arange(batch_size), np.argmax(assignment[:, :, 0], axis=1), 0
            ] = 1
            one_hot[
                np.arange(batch_size), np.argmax(assignment[:, :, 1], axis=1), 1
            ] = 1
        if np.std(regression) < 1:
            print(
                "Warning: Regression output seems to be constant. Check your model or data."
            )
        return one_hot, regression

    def evaluate_accuracy(self, data=None, truth=None, exclusive=True):
        """
        Evaluate the accuracy of the model's predictions.
        This method calculates the assignment accuracy and regression accuracy
        based on the provided data and truth values. If no data or truth values
        are provided, it uses the test data stored in the model.
        Args:
            data (dict, optional): A dictionary containing the input data with the
                following keys:
                - 'jet': Jet features.
                - 'lepton': Lepton features.
                - 'labels': Ground truth labels for assignment.
                - 'event_weight' (optional): Event weights for the data.
                - 'global' (optional): Global features, required if global features
                  are enabled in the model.
            truth (array-like, optional): Ground truth values for regression. If not
                provided, the method uses `self.y_test`.
            exclusive (bool, optional): Whether to use exclusive predictions. Defaults
                to True.
        Returns:
            tuple: A tuple containing:
                - Assignment accuracy metrics (as returned by `compute_assignment_accuracy`).
                - Regression accuracy metric (as returned by `compute_regression_accuracy`).
        Raises:
            ValueError: If only one of `data` or `truth` is provided.
            ValueError: If no test data is available and neither `data` nor `truth` is provided.
            ValueError: If the model has not been built.
            ValueError: If required keys ('jet', 'lepton', 'labels') are missing in `data`.
            ValueError: If 'global' key is missing in `data` when global features are enabled.
            ValueError: If the shapes of assignment predictions and truth values do not match.
        """

        if (truth is None) ^ (data is None):
            raise ValueError("Either both data and truth must be provided or neither.")
        if data is None and truth is None:
            if self.X_test is None or self.y_test is None:
                raise ValueError(
                    "No test data available. Please load or provide test data."
                )
            data = {
                "jet": self.X_test["jet"],
                "lepton": self.X_test["lepton"],
                "labels": self.X_test["labels"],
                "event_weight": (
                    self.X_test["event_weight"]
                    if "event_weight" in self.X_test
                    else None
                ),
                "global": self.X_test["global"] if self.global_features else None,
            }
        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )
        if "labels" not in data or "jet" not in data or "lepton" not in data:
            raise ValueError("Data must contain 'jet', 'lepton', and 'labels' keys.")
        if self.global_features and "global" not in data:
            raise ValueError(
                "Data must contain 'global' key when global features are used."
            )
        assignment_pred, regression_pred = self.make_prediction(
            data, exclusive=exclusive
        )
        assignment_truth = data["labels"]
        regression_truth = self.y_test if truth is None else truth
        if assignment_pred.shape != assignment_truth.shape:
            raise ValueError("Assignment prediction and truth shapes do not match.")
        return *self.compute_assignment_accuracy(
            assignment_truth, assignment_pred
        ), self.compute_regression_accuracy(regression_truth, regression_pred)

    @staticmethod
    def compute_assignment_accuracy(truth, prediction):
        """
        Compute the accuracy of assignments for two predicted labels compared to their ground truth values.
        Parameters:
        ----------
        truth : numpy.ndarray
            A 2D array where each row contains the ground truth labels for two elements (e.g., [lep_1_truth, lep_2_truth]).
        prediction : numpy.ndarray
            A 2D array where each row contains the predicted labels for two elements (e.g., [lep_1_pred, lep_2_pred]).
        Returns:
        -------
        tuple
            A tuple containing three accuracy metrics:
            - lep_1_accuracy (float): The accuracy of the first element's predictions.
            - lep_2_accuracy (float): The accuracy of the second element's predictions.
            - combined_accuracy (float): The accuracy of both elements being correctly predicted simultaneously.
        """

        lep_1_truth, lep_2_truth = truth[:, 0], truth[:, 1]
        lep_1_pred, lep_2_pred = prediction[:, 0], prediction[:, 1]

        lep_1_accuracy = np.mean(lep_1_truth == lep_1_pred)
        lep_2_accuracy = np.mean(lep_2_truth == lep_2_pred)
        combined_accuracy = np.mean(
            (lep_1_truth == lep_1_pred) & (lep_2_truth == lep_2_pred)
        )

        return lep_1_accuracy, lep_2_accuracy, combined_accuracy

    @staticmethod
    def compute_regression_accuracy(truth, prediction):
        """
        Compute the regression accuracy as the mean absolute percentage error (MAPE).
        Parameters:
            truth (array-like): The ground truth values.
            prediction (array-like): The predicted values.
        Returns:
            float: The mean absolute percentage error (MAPE) computed as 
                   the mean of the absolute differences between truth and 
                   prediction, normalized by the truth values.
        """
        
        return np.mean(np.abs(truth - prediction) / truth, axis=0)

    def plot_confusion_matrix(self, n_bootstrap=100, exclusive=True):
        """
        Plots the confusion matrix for the model's predictions on the test dataset.
        Parameters:
        -----------
        n_bootstrap : int, optional
            The number of bootstrap resampling iterations to perform for uncertainty estimation. 
            Default is 100.
        exclusive : bool, optional
            If True, ensures exclusive assignment during prediction. Default is True.
        Returns:
        --------
        matplotlib.figure.Figure
            A figure object representing the confusion matrix plot.
        Notes:
        ------
        - This method uses the `make_prediction` method to generate predictions for the test dataset.
        - The confusion matrix is generated using the `plot_external_confusion_matrix` method.
        - The `event_weight` column in `X_test` is used as sample weights for the confusion matrix.
        """
        
        assignment_pred, _ = self.make_prediction(self.X_test, exclusive=exclusive)
        truth_index = np.argmax(self.y_test, axis=1)
        max_jets = self.max_jets
        return self.plot_external_confusion_matrix(
            assignment_pred,
            truth_index,
            max_jets,
            n_bootstrap=n_bootstrap,
            sample_weight=self.X_test["event_weight"],
        )

    def get_confusion_matrix(self, exclusive=True):
        """
        Computes the confusion matrices for two lepton assignments based on the model's predictions.
        Args:
            exclusive (bool, optional): If True, ensures exclusive assignment of predictions. 
                Defaults to True.
        Returns:
            tuple: A tuple containing two confusion matrices:
                - lep_1_confusion_matrix (numpy.ndarray): Confusion matrix for the first lepton.
                - lep_2_confusion_matrix (numpy.ndarray): Confusion matrix for the second lepton.
        Notes:
            - The confusion matrices are normalized such that each row sums to 1.
            - The `self.X_test` dictionary is expected to contain the following keys:
                - "labels": Ground truth labels for the test data.
                - "event_weight": Weights for each event in the test data.
            - The `self.max_jets` attribute defines the number of possible jet assignments.
        """

        assignment_pred, _ = self.make_prediction(self.X_test, exclusive=exclusive)
        jet_matcher_indices = np.argmax(assignment_pred, axis=1)

        truth_index = np.argmax(self.X_test["labels"], axis=1)
        max_jets = self.max_jets
        lep_1_confusion_matrix = (
            confusion_matrix(
                truth_index[:, 0],
                jet_matcher_indices[:, 0],
                labels=np.arange(max_jets),
                sample_weight=self.X_test["event_weight"],
            ).astype("float")
            / confusion_matrix(
                truth_index[:, 0],
                jet_matcher_indices[:, 0],
                labels=np.arange(max_jets),
                sample_weight=self.X_test["event_weight"],
            ).sum(axis=1)[:, np.newaxis]
        )
        lep_2_confusion_matrix = (
            confusion_matrix(
                truth_index[:, 1],
                jet_matcher_indices[:, 1],
                labels=np.arange(max_jets),
                sample_weight=self.X_test["event_weight"],
            ).astype("float")
            / confusion_matrix(
                truth_index[:, 1],
                jet_matcher_indices[:, 1],
                labels=np.arange(max_jets),
                sample_weight=self.X_test["event_weight"],
            ).sum(axis=1)[:, np.newaxis]
        )
        return lep_1_confusion_matrix, lep_2_confusion_matrix

    @staticmethod
    def plot_external_confusion_matrix(
        jet_matcher_indices=None,
        truth_index=None,
        max_jets=None,
        n_bootstrap=100,
        sample_weight=None,
    ):
        """
        Plots confusion matrices for two leptons with bootstrap resampling.
        This function generates confusion matrices for two leptons (lepton 1 and lepton 2)
        using bootstrap resampling. It calculates the mean and standard deviation of the
        confusion matrices over multiple bootstrap samples and visualizes them using heatmaps.
        Parameters:
        -----------
        jet_matcher_indices : numpy.ndarray
            Array of predicted jet indices for each lepton. Shape should be (n_samples, 2),
            where the second dimension corresponds to the two leptons.
        truth_index : numpy.ndarray
            Array of true jet indices for each lepton. Shape should be (n_samples, 2),
            where the second dimension corresponds to the two leptons.
        max_jets : int
            The maximum number of jet categories (labels) to consider in the confusion matrix.
        n_bootstrap : int, optional, default=100
            The number of bootstrap resampling iterations to perform.
        sample_weight : numpy.ndarray, optional, default=None
            Array of sample weights for each data point. If None, all samples are weighted equally.
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The matplotlib figure object containing the confusion matrix plots.
        ax : numpy.ndarray of matplotlib.axes._subplots.AxesSubplot
            Array of axes objects corresponding to the two confusion matrix plots.
        Notes:
        ------
        - The function normalizes the confusion matrices by row (true label) to represent
          proportions.
        - The baseline confusion matrices (without resampling) are also plotted for comparison.
        - Standard deviations of the bootstrap confusion matrices are annotated on the heatmaps
          in red text.
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

    @staticmethod
    def compute_binned_regression_accuracy(
        truth: np.ndarray,
        prediction: np.ndarray,
        feature_data: np.ndarray,
        bins=20,
        xlims=None,
        event_weights: np.ndarray = None,
    ):
        """
        Compute the binned regression accuracy for a given set of predictions and truth values.
        This function calculates the binned weighted resolution, mean deviation, and accuracy 
        for regression predictions, grouped by bins of a specified feature. It also computes 
        the histogram of the feature data.
        Args:
            truth (np.ndarray): Ground truth values. Shape: (n_samples, n_features).
            prediction (np.ndarray): Predicted values. Shape: (n_samples, n_features).
            feature_data (np.ndarray): Feature data used for binning. Shape: (n_samples,).
            bins (int or sequence, optional): Number of bins or bin edges for the histogram. 
                Default is 20.
            xlims (tuple, optional): Range (min, max) for the feature data. If None, the range 
                is determined automatically. Default is None.
            event_weights (np.ndarray, optional): Weights for each event. Shape: (n_samples,). 
                If None, all events are equally weighted. Default is None.
        Returns:
            tuple: A tuple containing:
                - binned_accuracy_combined (np.ndarray): Standard deviation of the weighted 
                  resolution in each bin. Shape: (n_features, num_bins).
                - mean (np.ndarray): Mean of the weighted resolution in each bin. 
                  Shape: (n_features, num_bins).
                - binned_mean_deviation (np.ndarray): Mean deviation of the weighted resolution 
                  in each bin. Shape: (n_features, num_bins).
                - feature_bins (np.ndarray): Bin edges of the feature data. Shape: (num_bins + 1,).
                - feature_hist (np.ndarray): Histogram of the feature data. Shape: (num_bins,).
        """

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
        data_length = prediction.shape[0]
        number_features = prediction.shape[1]
        xlims = (feature_bins[0], feature_bins[-1])

        event_weights = (
            event_weights.reshape(-1, 1, 1)
            if event_weights is not None
            else np.ones((data_length, 1, 1))
        )
        event_weights = np.broadcast_to(
            event_weights, (data_length, number_features, 1)
        )

        binning_mask = (
            (feature_data.reshape(-1, 1, 1) < feature_bins[1:].reshape(1, 1, -1))
            & (feature_data.reshape(-1, 1, 1) >= feature_bins[:-1].reshape(1, 1, -1))
        ).astype(float)
        binned_weighted_resolution = (np.abs((truth - prediction) / truth)).reshape(
            data_length, number_features, 1
        ) * binning_mask

        binned_mean_deviation = np.sum(
            ((truth - prediction) / truth).reshape(data_length, number_features, 1)
            * binning_mask
            * event_weights,
            axis=0,
        ) / (np.sum(event_weights * binning_mask, axis=0) + 1e-9)

        weights = event_weights
        weighted_vals = binned_weighted_resolution * weights
        weighted_sq_vals = binned_weighted_resolution**2 * weights

        sum_weights = np.sum(
            weights * binning_mask, axis=0
        )  # (number_features, num_bins)
        mean = np.divide(
            np.sum(weighted_vals, axis=0),
            sum_weights,
            out=np.zeros_like(sum_weights),
            where=sum_weights != 0,
        )
        mean_sq = np.divide(
            np.sum(weighted_sq_vals, axis=0),
            sum_weights,
            out=np.zeros_like(sum_weights),
            where=sum_weights != 0,
        )

        variance = mean_sq - mean**2
        binned_accuracy_combined = np.sqrt(
            np.clip(variance, 0, None)
        )  # (number_features, num_bins)
        return (
            binned_accuracy_combined,
            mean,
            binned_mean_deviation,
            feature_bins,
            feature_hist,
        )

    def get_binned_regression_accuracy(
        self, feature_name, data_type="non_training", bins=20, xlims=None
    ):
        """
        Compute binned accuracy for a feature.

        Args:
            feature_name: str, name of the feature (for axis labeling)
            xlims: tuple (min, max) for feature range (optional)
            bins: int, number of bins

        Returns:
            binned_regression_accuracy: np.ndarray[], binned regression accuracy
        """
        if data_type not in self.feature_index_dict:
            raise ValueError(
                f"Data type '{data_type}' not found in feature index dictionary.\n Available types: {list(self.feature_index_dict.keys())}"
            )
        if feature_name not in self.feature_index_dict[data_type]:
            raise ValueError(
                f"Feature '{feature_name}' not found in data type '{data_type}'.\n Available features: {self.feature_index_dict[data_type]}"
            )
        feature_index = self.feature_index_dict[data_type][feature_name]
        if self.X_test is None or self.y_test is None:
            raise ValueError(
                "No test data available. Please load or provide test data."
            )
        feature_data = self.X_test[data_type][:, feature_index]

        assignment_pred, regression_pred = self.make_prediction(self.X_test)
        regression_truth = self.y_test

        return RegressionBaseModel.compute_binned_regression_accuracy(
            regression_truth,
            regression_pred,
            feature_data,
            bins=bins,
            xlims=xlims,
            event_weights=(
                self.X_test["event_weight"] if "event_weight" in self.X_test else None
            ),
        )

    def compute_permutation_importance(self, shuffle_number=1):
        """
        Compute permutation importance for each feature in the model.
        This method shuffles each feature in the test set and evaluates the model's performance.
        It returns the mean and standard deviation of the accuracy drop for each feature.
        Args:
            shuffle_number: int, number of times to shuffle each feature
        Returns:
            assignment_accuracies_mean: pd.DataFrame, mean and std of assignment accuracy drop for each feature
            regression_accuracies_dict: list of pd.DataFrame containing mean and std of regression accuracy drop for each feature for each target
        """
        if self.X_test is None or self.y_test is None:
            raise ValueError(
                "No test data available. Please load or provide test data."
            )
        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )


        _, _, assignment_baseline, regression_baseline = self.evaluate_accuracy(
            self.X_test, self.y_test
        )
        regression_accuracies = np.zeros((shuffle_number, self.n_regression_targets))
        assignment_accuracies = np.zeros((shuffle_number, 1))

        feature_names = (
            self.data_preprocessor.jet_features
            + self.data_preprocessor.lepton_features
            + self.data_preprocessor.global_features
            if self.global_features
            else []
        )

        regression_accuracies_dict = [
            pd.DataFrame(index=feature_names, columns=["mean", "std"])
            for _ in range(self.n_regression_targets)
        ]
        assignment_accuracies_mean = pd.DataFrame(
            index=feature_names, columns=["mean", "std"]
        )
        
        for feature in feature_names:
            X_test_permutated = deepcopy(self.X_test)
            y_test = self.y_test

            for data_type in ["jet", "lepton", "global"]:
                if feature in self.feature_index_dict[data_type]:
                    feature_index = self.feature_index_dict[data_type][feature]
                    for i in range(shuffle_number):
                        mask = X_test_permutated[data_type][:, :, feature_index] != self.padding_value
                        permuted_values = np.random.permutation(
                            X_test_permutated[data_type][:, :, feature_index][mask]
                        )
                        X_test_permutated[data_type][:, :, feature_index][mask] = permuted_values

                        _, _, assignment_accuracy, regression_accuracy = (
                            self.evaluate_accuracy(X_test_permutated, y_test)
                        )

                        regression_accuracies[i] = regression_accuracy
                        assignment_accuracies[i] = assignment_accuracy
                    break

            assignment_accuracies_mean.loc[feature, "mean"] = np.mean(
                assignment_baseline - assignment_accuracies
            )
            assignment_accuracies_mean.loc[feature, "std"] = np.std(
                assignment_baseline - assignment_accuracies
            )
        for i in range(self.n_regression_targets):
            regression_accuracies_dict[i].loc[feature, "mean"] = np.mean(
                (regression_accuracies[i] - regression_baseline[i])
            )
            regression_accuracies_dict[i].loc[feature, "std"] = np.std(
                (regression_accuracies[i] - regression_baseline[i])
            )
        return (
            assignment_accuracies_mean,
            regression_accuracies_dict,
        )

    def get_outputs(self):
        """
        Returns the outputs of the model.
        """
        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )
        if self.X_test is None or self.y_test is None:
            raise ValueError(
                "No test data available. Please load or provide test data."
            )
        assignment_pred, regression_pred = self.make_prediction(self.X_test)
        return assignment_pred, regression_pred, self.X_test["labels"], self.y_test
