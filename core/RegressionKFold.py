from .RegressionBaseModel import RegressionBaseModel
from .DataLoader import DataLoader, DataPreprocessor

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import threading
import copy


class RegressionKFold:
    """
    A class to handle k-fold cross-validation for regression models.
    It initializes multiple instances of a regression model type, loads data for each fold, and provides methods to build, compile, train,
    and evaluate the models in parallel.

    Args:
        model_type (type): The type of regression model to be used, which should inherit from RegressionBaseModel.
        data_processor (DataPreprocessor): An instance of DataPreprocessor to handle data processing and k-fold creation.
        n_folds (int): The number of folds for cross-validation. Default is 5.
        n_splits (int): The number of splits for each fold. Default is 1.
        random_state (int): Random seed for reproducibility. Default is 42.

    Attributes:
        model_type (type): The type of regression model to be used.
        feature_index_dict (dict): A dictionary containing feature indices for different data types.
        max_jets (int): The maximum number of jets in the dataset.
        max_leptons (int): The maximum number of leptons in the dataset.
        n_regression_targets (int): Number of regression targets.
        n_data_sets (int): The total number of datasets created from k-folds and splits.
        random_state (int): Random seed for reproducibility.
        model_list (list): A list of initialized regression model instances.
        k_fold (list): A list of tuples containing training and validation data for each fold.
        feature_data (dict): Feature data from the data processor.
    """

    def __init__(
        self,
        model_type: type[RegressionBaseModel],
        data_processor: DataPreprocessor,
        n_folds=5,
        n_splits=1,
        random_state=42,
    ):
        """
        Initialize the RegressionKFold object, create k-fold splits, and initialize models for each fold.

        Args:
            model_type (type): Regression model class.
            data_processor (DataPreprocessor): Data processor instance.
            n_folds (int): Number of folds.
            n_splits (int): Number of splits per fold.
            random_state (int): Random seed.
        """
        self.model_type: type[RegressionBaseModel] = model_type
        self.feature_index_dict: dict[str : dict[str:any]] = (
            data_processor.feature_index_dict
        )
        self.max_jets: int = data_processor.max_jets
        self.max_leptons: int = data_processor.max_leptons
        self.n_regression_targets: int = data_processor.n_regression_targets
        self.n_data_sets: int = n_folds * n_splits
        self.random_state: int = random_state
        self.model_list: list[RegressionBaseModel] = []
        self.k_fold = data_processor.create_k_folds(
            n_folds=n_folds, n_splits=n_splits, random_state=random_state
        )
        self.feature_data = data_processor.feature_data

        # Initialize a model for each fold/split
        for i in range(self.n_data_sets):
            self.model_list.append(model_type(data_processor))

        def load_data_thread(model: RegressionBaseModel, data):
            """
            Thread target for loading data into a model.
            """
            model.load_data(*data)

        # Load data for each model in parallel
        threads = []
        for i in range(self.n_data_sets):
            thread = threading.Thread(
                target=load_data_thread, args=(self.model_list[i], self.k_fold[i])
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def build_models(self, **kwargs):
        """
        Build all models in parallel using provided keyword arguments.
        """

        def build_model_thread(model: RegressionBaseModel, **kwargs):
            """
            Thread target for building a model.
            """
            model.build_model(**copy.deepcopy(kwargs))

        threads = []
        for model in self.model_list:
            thread = threading.Thread(
                target=build_model_thread, args=(model,), kwargs=kwargs
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def compile_models(self, **kwargs):
        """
        Compile all models in parallel using provided keyword arguments.
        """

        def compile_model_thread(model, **kwargs):
            """
            Thread target for compiling a model.
            """
            model.compile_model(**copy.deepcopy(kwargs))

        threads = []
        for model in self.model_list:
            thread = threading.Thread(
                target=compile_model_thread, args=(model,), kwargs=kwargs
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def compute_sample_weights(self, **kwargs):
        """
        Compute sample weights for all models in parallel.

        Args:
            **kwargs: Additional arguments for computing sample weights.
        """

        def compute_sample_weights_thread(model: RegressionBaseModel, **kwargs):
            """
            Thread target for computing sample weights.
            """
            model.compute_sample_weights(**copy.deepcopy(kwargs))

        threads = []
        for model in self.model_list:
            thread = threading.Thread(
                target=compute_sample_weights_thread, args=(model,), kwargs=kwargs
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def train_models(self, epochs=1, batch_size=1028, **kwargs):
        """
        Train all models in parallel.

        Args:
            epochs (int): Number of epochs.
            batch_size (int): Batch size.
            **kwargs: Additional arguments for training.
        """

        def train_model_thread(
            model: RegressionBaseModel, epochs, batch_size, **kwargs
        ):
            """
            Thread target for training a model.
            """
            model.train_model(
                epochs=epochs, batch_size=batch_size, **copy.deepcopy(kwargs)
            )

        threads = []
        for model in self.model_list:
            thread = threading.Thread(
                target=train_model_thread,
                args=(model, epochs, batch_size),
                kwargs=kwargs,
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def print_model_summary(self):
        """
        Print the summary of the first model.
        """
        self.model_list[0].summary()

    def enhance_region(
        self, variable, data_type, low_cut=None, high_cut=None, factor=1
    ):
        """
        Enhance a specific region of the data for all models.

        Args:
            variable (str): Variable name.
            data_type (str): Data type.
            low_cut (float): Lower cut value.
            high_cut (float): Upper cut value.
            factor (float): Enhancement factor.
        """
        for i in range(self.n_data_sets):
            self.model_list[i].enhance_region(
                variable, data_type, low_cut, high_cut, factor
            )

    def evaluate_accuracy(self):
        """
        Evaluate accuracy for all models in parallel.

        Returns:
            tuple: Means and errors for lepton 1, lepton 2, combined, and regression accuracies.
        """
        lep_1_accuracy = np.zeros(self.n_data_sets)
        lep_2_accuracy = np.zeros(self.n_data_sets)
        combined_accuracy = np.zeros(self.n_data_sets)
        regression_accuracy = np.zeros(
            (self.n_data_sets, self.n_regression_targets), dtype=float
        )

        def evaluate_accuracy_thread(
            model: RegressionBaseModel,
            index,
            lep_1_acc,
            lep_2_acc,
            combined_acc,
            regression_accuracy,
        ):
            """
            Thread target for evaluating accuracy of a model.
            """
            (
                lep_1_acc[index],
                lep_2_acc[index],
                combined_acc[index],
                regression_accuracy[index, :],
            ) = model.evaluate_accuracy()

        threads = []
        for i in range(self.n_data_sets):
            thread = threading.Thread(
                target=evaluate_accuracy_thread,
                args=(
                    self.model_list[i],
                    i,
                    lep_1_accuracy,
                    lep_2_accuracy,
                    combined_accuracy,
                    regression_accuracy,
                ),
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Calculate means and errors
        lep_1_accuracy_mean = np.mean(lep_1_accuracy)
        lep_2_accuracy_mean = np.mean(lep_2_accuracy)
        combined_accuracy_mean = np.mean(combined_accuracy)
        lep_1_accuracy_err = np.std(lep_1_accuracy) / np.sqrt(self.n_data_sets)
        lep_2_accuracy_err = np.std(lep_2_accuracy) / np.sqrt(self.n_data_sets)
        combined_accuracy_err = np.std(combined_accuracy) / np.sqrt(self.n_data_sets)
        regression_accuracy_mean = np.mean(regression_accuracy, axis=0)
        regression_accuracy_err = np.std(regression_accuracy, axis=0) / np.sqrt(
            self.n_data_sets
        )
        return (
            lep_1_accuracy_mean,
            lep_1_accuracy_err,
            lep_2_accuracy_mean,
            lep_2_accuracy_err,
            combined_accuracy_mean,
            combined_accuracy_err,
            regression_accuracy_mean,
            regression_accuracy_err,
        )

    def plot_confusion_matrix(self, exclusive=True):
        """
        Plot the mean and standard deviation of confusion matrices for all models.

        Args:
            exclusive (bool): Whether to use exclusive confusion matrix.

        Returns:
            tuple: Figure and axes of the plot.
        """
        confusion_matrices_lep_1 = [None] * self.n_data_sets
        confusion_matrices_lep_2 = [None] * self.n_data_sets
        max_jets = self.max_jets

        def get_confusion_matrix_thread(
            model: RegressionBaseModel, index, exclusive, results_lep_1, results_lep_2
        ):
            """
            Thread target for getting confusion matrices from a model.
            """
            confusion_matrix_lep_1, confusion_matrix_lep_2 = model.get_confusion_matrix(
                exclusive=exclusive
            )
            results_lep_1[index] = confusion_matrix_lep_1
            results_lep_2[index] = confusion_matrix_lep_2

        threads = []
        for i in range(self.n_data_sets):
            thread = threading.Thread(
                target=get_confusion_matrix_thread,
                args=(
                    self.model_list[i],
                    i,
                    exclusive,
                    confusion_matrices_lep_1,
                    confusion_matrices_lep_2,
                ),
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
        # Calculate mean and std for confusion matrices
        mean_confusion_lep_1 = np.mean(confusion_matrices_lep_1, axis=0)
        mean_confusion_lep_2 = np.mean(confusion_matrices_lep_2, axis=0)
        std_confusion_lep_1 = np.std(confusion_matrices_lep_1, axis=0) / np.sqrt(
            self.n_data_sets
        )
        std_confusion_lep_2 = np.std(confusion_matrices_lep_2, axis=0) / np.sqrt(
            self.n_data_sets
        )
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        sns.heatmap(
            mean_confusion_lep_1,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar=False,
            ax=ax[0],
        )
        sns.heatmap(
            mean_confusion_lep_2,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar=False,
            ax=ax[1],
        )
        # Annotate with std
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

    def save_model(self, file_path="model.keras"):
        """
        Save all models to disk in parallel.

        Args:
            file_path (str): Base file path for saving models.
        """

        def save_model_thread(model, file_path):
            """
            Thread target for saving a model.
            """
            model.save_model(file_path)

        threads = []
        for i in range(self.n_data_sets):
            thread = threading.Thread(
                target=save_model_thread,
                args=(self.model_list[i], file_path.replace(".keras", f"_{i}.keras")),
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
        print(f"Models saved to {file_path}.")

    def load_model(self, file_path="model.keras"):
        """
        Load all models from disk.

        Args:
            file_path (str): Base file path for loading models.
        """
        for i in range(self.n_data_sets):
            file_path_i = file_path.replace(".keras", f"_{i}.keras")
            print("Loading model from", file_path_i)
            self.model_list[i].load_model(file_path_i)
        print(f"Models loaded from {file_path}.")

    def plot_history(self):
        """
        Plot training history for all models.

        Returns:
            tuple: Lists of figures and axes for each model.
        """
        fig, ax = [], []
        for i in range(self.n_data_sets):
            fig_i, ax_i = self.model_list[i].plot_history()
            fig.append(fig_i)
            ax.append(ax_i)
        return fig, ax

    def plot_feature_variance(
        self, feature_name, data_type="non_training", bins=50, xlims=None
    ):
        """
        Plot the variance of a feature across all folds.

        Args:
            feature_name (str): Name of the feature.
            data_type (str): Data type.
            bins (int): Number of bins.
            xlims (tuple): Limits for the x-axis.

        Returns:
            tuple: Figure and axis of the plot.
        """
        if data_type not in self.feature_index_dict:
            raise ValueError(
                f"Data type {data_type} not found. Use 'jet', 'lepton', 'non_training', or 'global'."
            )
        if feature_name not in self.feature_index_dict[data_type]:
            raise ValueError(
                f"Feature {feature_name} not found in {data_type} features."
            )
        # Compute histogram bins using all data
        _, hist_bins = np.histogram(
            np.concatenate(
                [
                    self.model_list[i].X_test[data_type][
                        :,
                        self.model_list[i].feature_index_dict[data_type][feature_name],
                    ]
                    for i in range(self.n_data_sets)
                ]
            ),
            bins=bins,
            range=xlims,
        )
        feature_histos = np.zeros((self.n_data_sets, bins))
        # Compute histogram for each fold
        for i in range(self.n_data_sets):
            feature_histos[i], _ = np.histogram(
                self.model_list[i].X_test[data_type][
                    :, self.model_list[i].feature_index_dict[data_type][feature_name]
                ],
                bins=hist_bins,
            )
        feature_histos_min = np.min(feature_histos, axis=0)
        feature_histos_max = np.max(feature_histos, axis=0)
        feature_histos_mean = np.mean(feature_histos, axis=0)
        fig, ax = plt.subplots(figsize=(10, 5))
        centers = (hist_bins[:-1] + hist_bins[1:]) / 2
        # Plot range as bar
        ax.bar(
            centers,
            feature_histos_max - feature_histos_min,
            bottom=feature_histos_min,
            width=np.diff(hist_bins),
            alpha=0.5,
            color="orange",
            label="Feature Count Range",
        )
        # Plot mean as line
        for i in range(bins):
            ax.plot(
                [hist_bins[i], hist_bins[i + 1]],
                [feature_histos_mean[i], feature_histos_mean[i]],
                color="black",
            )
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Feature Count")
        ax.set_title("Feature Variance Across K-Folds")
        fig.tight_layout()
        return fig, ax

    def plot_binned_regression_accuracy(
        self,
        feature_name,
        data_type="non_training",
        bins=20,
        xlims=None,
        additional_data_to_plot=None,
    ):
        """
        Plot binned regression accuracy as a function of a feature.

        Args:
            feature_name (str): Name of the feature.
            data_type (str): Data type.
            bins (int): Number of bins.
            xlims (tuple): Limits for the x-axis.

        Returns:
            tuple: Figure and axes of the plot.
        """
        if data_type not in self.feature_index_dict:
            raise ValueError(
                f"Data type {data_type} not found. Use 'jet', 'lepton', 'non_training', or 'global'."
            )
        if feature_name not in self.feature_index_dict[data_type]:
            raise ValueError(
                f"Feature {feature_name} not found in {data_type} features. \n Available features: {self.feature_index_dict[data_type].keys()}"
            )

        # Compute histogram for the feature
        feature_hist, feature_bins = np.histogram(
            self.feature_data[data_type][
                :, self.feature_index_dict[data_type][feature_name]
            ],
            bins=bins,
            range=xlims,
        )

        def compute_binned_accuracy_thread(
            model: RegressionBaseModel,
            index,
            feature_name,
            data_type,
            bins,
            binned_accuracy_combined,
            binned_abs_mean_combined,
            binned_mean_deviation_combined,
        ):
            """
            Thread target for computing binned regression accuracy.
            """
            (
                binned_accuracy_combined[index],
                binned_abs_mean_combined[index],
                binned_mean_deviation_combined[index],
                _,
                _,
            ) = model.get_binned_regression_accuracy(
                feature_name,
                data_type=data_type,
                bins=bins,
            )

        if additional_data_to_plot is not None:
            if not isinstance(additional_data_to_plot, dict):
                raise ValueError("additional_data_to_plot must be a dictionary.")
            for key, value in additional_data_to_plot.items():
                if not isinstance(value, tuple) or (
                    len(value) != 3 and len(value) != 4
                ):
                    raise ValueError(f"Value for {key} must be a tuple of (x, y).")
                if (
                    not isinstance(value[0], np.ndarray)
                    or not isinstance(value[1], np.ndarray)
                    or not isinstance(value[2], np.ndarray)
                ):
                    raise ValueError(
                        f"Both elements of the tuple for {key} must be numpy arrays."
                    )
                if len(value) == 4 and not isinstance(value[3], np.ndarray):
                    raise ValueError(
                        f"Optional event weights for {key} must be a numpy array."
                    )
                if len(value) == 4 and value[3].shape != value[0].shape:
                    raise ValueError(
                        f"Event weights for {key} must have the same length as x and y arrays."
                    )
                if len(value[0]) != len(value[1]) or len(value[0]) != len(value[2]):
                    raise ValueError(
                        f"x and y arrays for {key} must have the same length."
                    )
            additional_plotting_data_dict = {}
            for key, value in additional_data_to_plot.items():
                (
                    binned_accuracy_combined,
                    binned_abs_mean_combined,
                    binned_mean_deviation_combined,
                    _,
                    _,
                ) = RegressionBaseModel.compute_binned_regression_accuracy(
                    value[0],
                    value[1],
                    value[2],
                    bins=feature_bins,
                    event_weights=value[3] if len(value) == 4 else None,
                )
                additional_plotting_data_dict[key] = {
                    "binned_accuracy": binned_accuracy_combined,
                    "binned_abs_mean": binned_abs_mean_combined,
                    "binned_mean_deviation": binned_mean_deviation_combined,
                }
        threads = []
        binned_accuracy_combined = np.zeros(
            (
                self.n_data_sets,
                self.n_regression_targets,
                bins,
            ),
            dtype=float,
        )
        binned_abs_mean_combined = np.zeros(
            (
                self.n_data_sets,
                self.n_regression_targets,
                bins,
            ),
            dtype=float,
        )
        binned_mean_deviation_combined = np.zeros(
            (
                self.n_data_sets,
                self.n_regression_targets,
                bins,
            ),
            dtype=float,
        )

        # Compute binned accuracy for each fold in parallel
        for i in range(self.n_data_sets):
            thread = threading.Thread(
                target=compute_binned_accuracy_thread,
                args=(
                    self.model_list[i],
                    i,
                    feature_name,
                    data_type,
                    feature_bins,
                    binned_accuracy_combined,
                    binned_abs_mean_combined,
                    binned_mean_deviation_combined,
                ),
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Calculate mean and error
        binned_accuracy_mean = np.mean(binned_accuracy_combined, axis=0)
        binned_accuracy_err = np.std(binned_accuracy_combined, axis=0) / np.sqrt(
            self.n_data_sets
        )

        binned_mean_deviation = np.mean(binned_mean_deviation_combined, axis=0)
        binned_mean_deviation_err = np.std(
            binned_mean_deviation_combined, axis=0
        ) / np.sqrt(self.n_data_sets)

        centers = (feature_bins[:-1] + feature_bins[1:]) / 2
        fig, ax = plt.subplots(
            ncols=self.n_regression_targets,
            nrows=2,
            figsize=(self.n_regression_targets * 8, 8),
            sharex="col",
        )
        if self.n_regression_targets == 1:
            ax = ax.reshape(2, 1)  # Reshape for single target case
        for i in range(self.n_regression_targets):
            ax_twin = ax[0, i].twinx()
            ax_twin.set_ylabel("Feature Count", color="tab:orange")
            ax_twin.bar(
                centers,
                feature_hist,
                width=np.diff(feature_bins),
                alpha=0.3,
                color="tab:orange",
            )
            ax_twin.tick_params(axis="y", labelcolor="tab:orange")
            ax[0, i].tick_params(axis="y")
            ax[0, i].errorbar(
                centers,
                binned_accuracy_mean[i],
                yerr=binned_accuracy_err[i],
                label=f"{self.model_type.__name__} Matcher",
                fmt="o",
                capsize=2,
            )
            ax[0, i].set_ylabel(
                r"Resolution $\sigma\left(\left\|\frac{\text{truth}-\text{reco}}{\text{truth}}\right\|\right)$"
            )
            ax[0, i].set_xlim(xlims)
            ylim = np.max(binned_accuracy_mean[i]) + 0.05
            if additional_data_to_plot is not None:
                for key, data in additional_plotting_data_dict.items():
                    ax[0, i].errorbar(
                        centers,
                        data["binned_accuracy"][i],
                        label=key,
                        fmt="o",
                        capsize=2,
                    )
                    if np.max(data["binned_accuracy"][i]) > ylim:
                        ylim = np.max(data["binned_accuracy"][i]) + 0.05
            ax[0, i].set_ylim(0, ylim)
            ax_twin = ax[1, i].twinx()
            ax_twin.set_ylabel("Feature Count", color="tab:orange")
            ax_twin.bar(
                centers,
                feature_hist,
                width=np.diff(feature_bins),
                alpha=0.3,
                color="tab:orange",
            )
            ax_twin.tick_params(axis="y", labelcolor="tab:orange")
            ax[1, i].tick_params(axis="y")
            ax[1, i].errorbar(
                centers,
                binned_mean_deviation[i],
                yerr=binned_mean_deviation_err[i],
                label=f"{self.model_type.__name__} Matcher",
                fmt="o",
                capsize=2,
            )
            y_lim_low = np.min(binned_mean_deviation[i]) - 0.05
            y_lim_high = np.max(binned_mean_deviation[i]) + 0.05
            if additional_data_to_plot is not None:
                for key, data in additional_plotting_data_dict.items():
                    ax[1, i].errorbar(
                        centers,
                        data["binned_mean_deviation"][i],
                        label=key,
                        fmt="o",
                        capsize=2,
                    )
                    if np.min(data["binned_mean_deviation"][i]) < y_lim_low:
                        y_lim_low = np.min(data["binned_mean_deviation"][i]) - 0.05
                    if np.max(data["binned_mean_deviation"][i]) > y_lim_high:
                        y_lim_high = np.max(data["binned_mean_deviation"][i]) + 0.05
            ax[1, i].set_ylim(y_lim_low, y_lim_high)
            ax[1, i].set_ylabel(
                r"Mean Deviation $\mu\left(\frac{\text{truth}-\text{reco}}{\text{truth}}\right)$"
            )
            ax[1, i].set_xlim(xlims)

            ax[1, i].set_xlabel(feature_name)
            ax[0, i].get_xaxis().set_visible(False)
        ax[0, 0].legend(loc="best")
        fig.tight_layout()
        return fig, ax

    def plot_permutation_importance(self):
        """
        Plot permutation importance for assignment and regression accuracy.

        Returns:
            tuple: Figure and axes of the plot.
        """
        assignment_accuracies, regression_accuracies = [None] * self.n_data_sets, [
            None
        ] * self.n_data_sets

        def compute_permutation_importance_thread(
            model: RegressionBaseModel,
            index,
            assignment_accuracies,
            regression_accuracies,
        ):
            """
            Thread target for computing permutation importance.
            """
            (
                assignment_accuracies[index],
                regression_accuracies[index],
            ) = model.compute_permutation_importance(shuffle_number=1)

        threads = []
        for i in range(self.n_data_sets):
            thread = threading.Thread(
                target=compute_permutation_importance_thread,
                args=(
                    self.model_list[i],
                    i,
                    assignment_accuracies,
                    regression_accuracies,
                ),
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Calculate mean and std for assignment and regression accuracies
        assignment_accuracies_mean = pd.concat(assignment_accuracies, axis=1).mean(
            axis=1
        )
        assignment_accuracies_std = pd.concat(assignment_accuracies, axis=1).std(axis=1)
        regression_accuracies_mean_list = []
        regression_accuracies_std_list = []
        for i in range(self.n_regression_targets):
            regression_accuracies_mean_list.append(
                pd.concat(
                    [regression_accuracies[j][i] for j in range(self.n_data_sets)],
                    axis=1,
                ).mean(axis=1)
            )
            regression_accuracies_std_list.append(
                pd.concat(
                    [regression_accuracies[j][i] for j in range(self.n_data_sets)],
                    axis=1,
                ).std(axis=1)
            )

        fig, ax = plt.subplots(
            nrows=self.n_regression_targets + 1,
            ncols=1,
            figsize=(10, 5 * (self.n_regression_targets + 1)),
        )
        # Plot regression target importances
        for i in range(self.n_regression_targets):
            regression_accuracies_mean_list[i] = regression_accuracies_mean_list[
                i
            ].sort_values(ascending=False)
            regression_accuracies_mean_list[i].plot(
                kind="bar",
                y="mean",
                yerr=regression_accuracies_std_list[i] / np.sqrt(self.n_data_sets),
                capsize=5,
                ax=ax[i],
                color="blue",
                alpha=0.7,
            )
            ax[i].set_title(f"Permutation Importance for Regression Target {i + 1}")
            ax[i].set_xlabel("Feature")
            ax[i].set_ylabel("Importance Score")

        # Plot assignment accuracy importances
        assignment_accuracies_mean = assignment_accuracies_mean.sort_values(
            ascending=False
        )
        assignment_accuracies_mean.plot(
            kind="bar",
            y="mean",
            yerr=assignment_accuracies_std / np.sqrt(self.n_data_sets),
            capsize=5,
            ax=ax[self.n_regression_targets],
            color="blue",
            alpha=0.7,
        )

        ax[self.n_regression_targets].set_title(
            "Permutation Importance for Assignment Accuracy"
        )
        ax[self.n_regression_targets].set_xlabel("Feature")
        ax[self.n_regression_targets].set_ylabel("Importance Score")
        fig.tight_layout()
        return fig, ax

    def plot_relative_variation_histogram(self, xlims=(-1, 1), bins=30):
        """
        Plots a histogram of the relative variation of regression targets across all folds.
        Args:
            xlims (tuple): Limits for the x-axis of the histogram.
            bins (int): Number of bins for the histogram.
        returns:
            tuple: Figure and axes of the plot.
        """
        relative_variations = np.zeros(
            (self.n_data_sets, self.n_regression_targets, bins), dtype=float
        )
        bin_edges = np.linspace(xlims[0], xlims[1], bins + 1)

        def compute_relative_variation_thread(
            model: RegressionBaseModel, index, relative_variations
        ):
            """
            Thread target for computing relative variation of regression targets.
            """
            _, regression_pred, _, regression_truth = model.get_outputs()
            for regression_target_index in range(self.n_regression_targets):
                relative_variations[index, regression_target_index, :], _ = (
                    np.histogram(
                        (
                            regression_pred[:, regression_target_index]
                            - regression_truth[:, regression_target_index]
                        )
                        / regression_truth[:, regression_target_index],
                        bins=bin_edges,
                    )
                )

        threads = []
        for i in range(self.n_data_sets):
            thread = threading.Thread(
                target=compute_relative_variation_thread,
                args=(self.model_list[i], i, relative_variations),
            )
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

        relative_variations = relative_variations / np.sum(
            relative_variations, axis=2, keepdims=True
        )

        relative_variations_mean = np.mean(relative_variations, axis=0)
        relative_variations_err = np.std(relative_variations, axis=0) / np.sqrt(
            self.n_data_sets
        )
        fig, ax = plt.subplots(
            nrows=self.n_regression_targets,
            ncols=1,
            figsize=(10, 5 * self.n_regression_targets),
        )
        if self.n_regression_targets == 1:
            ax = np.array([ax])
        for i in range(self.n_regression_targets):
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax[i].bar(
                centers,
                relative_variations_mean[i],
                yerr=relative_variations_err[i],
                width=np.diff(bin_edges),
                alpha=0.7,
                color="blue",
                label=f"{self.model_type.__name__} Matcher",
            )
            ax[i].set_xlabel("Relative Variation")
            ax[i].set_ylabel("Count")
            ax[i].set_title(
                f"Relative Variation Histogram for Regression Target {i + 1}"
            )
            ax[i].set_xlim(xlims)
            ax[i].set_ylim(0, np.max(relative_variations_mean[i]) + 0.05)
        fig.tight_layout()
        return fig, ax

    def plot_prediction_2d_hist(self, xlims=None, bins=10, q=0):
        """
        Plots a 2D histogram of predictions vs. truth for each regression target.

        Args:
            xlims (tuple or list): Limits for the x-axis of the histogram.
            bins (int): Number of bins for the histogram.
            q (float): Percentile for xlims calculation.

        Returns:
            tuple: Figure and axes of the plot.
        """
        if xlims is None:
            xlims = [
                (
                    np.percentile(self.feature_data["regression_targets"][:, i], q),
                    np.percentile(
                        self.feature_data["regression_targets"][:, i], 100 - q
                    ),
                )
                for i in range(self.n_regression_targets)
            ]
        elif isinstance(xlims, tuple):
            xlims = [xlims] * self.n_regression_targets
        elif not (isinstance(xlims, list) and len(xlims) == self.n_regression_targets):
            raise ValueError(
                "xlims must be a tuple or a list of tuples for each regression target."
            )

        histograms = np.zeros((self.n_data_sets, self.n_regression_targets, bins, bins))
        regression_predictions = []

        def compute_2d_histogram(model: RegressionBaseModel, index):
            _, pred, _, truth = model.get_outputs()
            regression_predictions.append(pred)
            for j in range(self.n_regression_targets):
                histograms[index, j], _, _ = np.histogram2d(
                    pred[:, j], truth[:, j], bins=bins, range=[xlims[j], xlims[j]]
                )

        threads = [
            threading.Thread(target=compute_2d_histogram, args=(model, i))
            for i, model in enumerate(self.model_list)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        regression_predictions = np.concatenate(regression_predictions, axis=0)
        histograms_mean = np.mean(histograms, axis=0)
        histograms_mean /= np.sum(histograms_mean, axis=2, keepdims=True) + 1e-9

        fig, axes = plt.subplots(
            1, self.n_regression_targets, figsize=(5 * self.n_regression_targets, 5)
        )
        if self.n_regression_targets == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            feature_name = next(
                (
                    name
                    for name, idx in self.feature_index_dict[
                        "regression_targets"
                    ].items()
                    if idx == i
                ),
                None,
            )
            if not feature_name:
                raise ValueError(f"Feature for regression target {i} not found.")

            centers = np.linspace(xlims[i][0], xlims[i][1], bins + 1)
            X, Y = np.meshgrid(
                (centers[:-1] + centers[1:]) / 2, (centers[:-1] + centers[1:]) / 2
            )

            pcm = ax.pcolormesh(
                X, Y, histograms_mean[i].T, shading="auto", cmap="Blues"
            )
            ax.set_xlabel(f"Predicted {feature_name}")
            ax.set_ylabel(f"True {feature_name}")
            ax.set_xlim(xlims[i])
            ax.set_ylim(xlims[i])

            # Add projections
            ax_pred = ax.inset_axes([0, 1.0, 1, 0.2], sharex=ax)
            ax_true = ax.inset_axes([1.0, 0, 0.2, 1], sharey=ax)
            ax_pred.hist(
                regression_predictions[:, i],
                bins=5 * bins,
                range=xlims[i],
                color="orange",
                alpha=0.5,
            )
            ax_true.hist(
                self.feature_data["regression_targets"][:, i],
                bins=5 * bins,
                range=xlims[i],
                orientation="horizontal",
                color="orange",
                alpha=0.5,
            )
            ax_pred.get_xaxis().set_visible(False)
            ax_true.get_yaxis().set_visible(False)
            ax_pred.get_yaxis().set_visible(False)
            ax_true.get_xaxis().set_visible(False)
            fig.colorbar(
                pcm,
                ax=ax_true,
                label="Normalized Count",
            )

        fig.tight_layout()
        return fig, axes
