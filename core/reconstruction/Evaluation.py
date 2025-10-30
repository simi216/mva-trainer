import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from typing import Union, Optional, Tuple, List
from copy import deepcopy

from . import EventReconstructorBase, MLReconstructorBase, GroundTruthReconstructor
from core.utils import (
    lorentz_vector_from_PtEtaPhiE_array,
    lorentz_vector_from_neutrino_momenta_array,
    compute_mass_from_lorentz_vector_array,
)


class MLEvaluator:
    """Base evaluator for ML-based jet assignment models."""

    def __init__(self, reconstructor: MLReconstructorBase, X_test, y_test):
        self.reconstructor = reconstructor
        self.X_test = X_test
        self.y_test = y_test
        self.max_leptons = reconstructor.max_leptons
        self.max_jets = reconstructor.max_jets
        self.met_features = reconstructor.met_features
        self.n_jets: int = reconstructor.n_jets
        self.n_leptons: int = reconstructor.n_leptons
        self.n_met: int = reconstructor.n_met
        self.padding_value: float = reconstructor.padding_value
        self.feature_indices = reconstructor.config.feature_indices

    def plot_training_history(self):
        """Plot training and validation loss/accuracy over epochs."""
        if self.reconstructor.history is None:
            raise ValueError(
                "No training history found. Please train the model before plotting history."
            )

        history = self.reconstructor.history

        # Plot loss
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(history.history["loss"], label="Training Loss")
        ax[0].plot(history.history["val_loss"], label="Validation Loss")
        ax[0].set_title("Model Loss")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].legend()
        # Plot accuracy
        ax[1].plot(history.history["assignment_accuracy"], label="Training Accuracy")
        ax[1].plot(
            history.history["val_assignment_accuracy"], label="Validation Accuracy"
        )
        ax[1].set_title("Model Accuracy")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Accuracy")
        ax[1].legend()
        return fig, ax

    def compute_permutation_importance(self, num_repeats: int = 5) -> dict:
        """
        Compute feature importance using permutation importance.

        Args:
            num_repeats: Number of times to permute each feature

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.X_test is None or self.y_test is None:
            raise ValueError(
                "Test data not loaded. Please load test data before computing permutation importance."
            )

        baseline_performance = self.reconstructor.evaluate_accuracy(
            self.X_test, self.y_test["assignment_labels"]
        )
        print(f"Baseline Performance: {baseline_performance:.4f}")

        importances = {}

        # Jet features
        for feature, feature_idx in self.config.feature_indices["jet"].items():
            scores = []
            for _ in range(num_repeats):
                X_permuted = deepcopy(self.X_test)
                mask = np.any(X_permuted["jet"] != self.padding_value, axis=-1)
                X_permuted["jet"][mask, feature_idx] = np.random.permutation(
                    X_permuted["jet"][mask, feature_idx]
                )
                permuted_performance = self.reconstructor.evaluate_accuracy(
                    X_permuted, self.y_test["assignment_labels"]
                )
                scores.append(baseline_performance - permuted_performance)
            importances[feature] = np.mean(scores)

        # Lepton features
        for feature, feature_idx in self.config.feature_indices["lepton"].items():
            scores = []
            for _ in range(num_repeats):
                X_permuted = deepcopy(self.X_test)
                X_permuted["lepton"][:, :, feature_idx] = np.random.permutation(
                    X_permuted["lepton"][:, :, feature_idx]
                )
                permuted_performance = self.reconstructor.evaluate_accuracy(
                    X_permuted, self.y_test["assignment_labels"]
                )
                scores.append(baseline_performance - permuted_performance)
            importances[feature] = np.mean(scores)

        # met features
        if self.met_features:
            for feature, feature_idx in self.config.feature_indices["met"].items():
                scores = []
                for _ in range(num_repeats):
                    X_permuted = deepcopy(self.X_test)
                    X_permuted["met"][:, :, feature_idx] = np.random.permutation(
                        X_permuted["met"][:, :, feature_idx]
                    )
                    permuted_performance = self.reconstructor.evaluate_accuracy(
                        X_permuted, self.y_test["assignment_labels"]
                    )
                    scores.append(baseline_performance - permuted_performance)
                importances[feature] = np.mean(scores)

        return importances

    def plot_feature_importance(self, num_repeats=10):
        """Plot feature importance scores."""
        importances = self.compute_permutation_importance(num_repeats=num_repeats)
        features = list(importances.keys())
        scores = list(importances.values())

        sorted_indices = np.argsort(scores)[::-1]
        features = [features[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(features, scores, color="skyblue")
        ax.set_xlabel("Importance Score")
        ax.set_title("Feature Importance based on Permutation Importance")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()


class ReconstructionEvaluator:
    """Evaluator for comparing event reconstruction methods."""

    def __init__(
        self,
        reconstructors: Union[list[EventReconstructorBase], EventReconstructorBase],
        X_test,
        y_test,
        neutrino_momenta_branches: Optional[List[str]] = [
            "nu_flows_neutriono_p_x",
            "nu_flows_neutriono_p_y",
            "nu_flows_neutriono_p_z",
            "nu_flows_anti_neutriono_p_x",
            "nu_flows_anti_neutriono_p_y",
            "nu_flows_anti_neutriono_p_z",
        ],
    ):
        if isinstance(reconstructors, EventReconstructorBase):
            self.reconstructors = [reconstructors]
        else:
            self.reconstructors = reconstructors

        self.X_test = X_test
        self.y_test = y_test
        self.model_predictions = []

        # Validate that all reconstructors have the same configuration
        configs = [reconstructor.config for reconstructor in self.reconstructors]
        select_config = configs[0]
        for config in configs[1:]:
            if config != select_config:
                raise ValueError(
                    "All reconstructors must have the same DataConfig for consistent evaluation."
                )
        self.config = select_config
        if neutrino_momenta_branches is None:
            for reconstructor in self.reconstructors:
                if not reconstructor.neutrino_reconstruction:
                    raise ValueError(
                        "Neutrino reconstruction is disabled for one or more reconstructors. Please provide neutrino_momenta_branches."
                    )
        else:
            if not all(
                [
                    branch in self.config.feature_indices["non_training"]
                    for branch in neutrino_momenta_branches
                ]
            ):
                raise ValueError(
                    "One or more neutrino_momenta_branches not found in config.feature_indices['non_training']."
                )
            else:
                self.neutrino_momenta = np.array(
                    [
                        self.X_test["non_training"][
                            :, self.config.feature_indices["non_training"][branch]
                        ]
                        for branch in neutrino_momenta_branches
                    ]
                ).T.reshape(-1, 2, 3)
        self.compute_all_model_predictions()

    def compute_all_model_predictions(self):
        """Compute predictions for all reconstructors."""
        for reconstructor in self.reconstructors:
            assignment_predictions = reconstructor.predict_indices(self.X_test)
            if self.config.has_regression_targets:
                regression_predictions = reconstructor.reconstruct_neutrinos(
                    self.X_test
                ).T.reshape(-1, 2, 3)
                self.model_predictions.append(
                    {
                        "assignment": assignment_predictions,
                        "regression": regression_predictions,
                    }
                )
            else:
                assignment_predictions = reconstructor.predict_indices(self.X_test)
                neutrino_predictions = self.neutrino_momenta
                self.model_predictions.append(
                    {
                        "assignment": assignment_predictions,
                        "regression": neutrino_predictions,
                    }
                )

    def evaluate_accuracy(
        self, true_labels, predictions, per_event: bool = True
    ) -> float:
        """
        Evaluates the accuracy of the model on the provided true labels and predictions.

        Args:
            true_labels (np.ndarray): The true assignment labels to compare against the model's predictions.
            predictions (np.ndarray): The model's predicted assignment probabilities.
            per_event (bool): If True, computes accuracy per event; otherwise, computes overall accuracy.
        Returns:

        """
        predicted_indices = np.argmax(predictions, axis=-2)
        true_indices = np.argmax(true_labels, axis=-2)
        if per_event:
            correct_predictions = np.all(predicted_indices == true_indices, axis=-1)
            accuracy = correct_predictions
        else:
            correct_predictions = np.all(predicted_indices == true_indices, axis=-1)
            accuracy = np.mean(correct_predictions)
        return accuracy

    def _bootstrap_accuracy(
        self,
        reconstructor_index: int,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> Tuple[float, float, float]:
        """
        Compute accuracy with bootstrap confidence intervals.

        Args:
            reconstructor_index: Index of the reconstructor in self.reconstructors
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level for intervals

        Returns:
            Tuple of (mean_accuracy, lower_bound, upper_bound)
        """
        predictions = self.model_predictions[reconstructor_index]["assignment"]
        accuracy_data = self.evaluate_accuracy(
            self.y_test["assignment_labels"], predictions, per_event=True
        ).astype(float)

        n_samples = len(accuracy_data)
        bootstrap_accuracies = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.randint(0, n_samples, size=n_samples)
            bootstrap_accuracy = accuracy_data[indices]
            bootstrap_accuracies[i] = np.mean(bootstrap_accuracy)

        mean_accuracy = np.mean(bootstrap_accuracies)

        # Compute confidence intervals
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bound = np.percentile(bootstrap_accuracies, lower_percentile)
        upper_bound = np.percentile(bootstrap_accuracies, upper_percentile)

        return mean_accuracy, lower_bound, upper_bound

    def plot_all_accuracies(
        self,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        figsize: Tuple[int, int] = (10, 6),
    ):
        """
        Plot accuracies for all reconstructors with error bars from bootstrapping.

        Args:
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level for error bars
            figsize: Figure size
        """
        names = []
        accuracies = []
        errors_lower = []
        errors_upper = []

        print("\nComputing bootstrap confidence intervals...")
        for reconstructor_index, reconstructor in enumerate(self.reconstructors):
            mean_acc, lower, upper = self._bootstrap_accuracy(
                reconstructor_index, n_bootstrap, confidence
            )
            names.append(reconstructor.get_name())
            accuracies.append(mean_acc)
            errors_lower.append(mean_acc - lower)
            errors_upper.append(upper - mean_acc)

            print(
                f"{reconstructor.get_name()}: {mean_acc:.4f} "
                f"[{lower:.4f}, {upper:.4f}]"
            )

        fig, ax = plt.subplots(figsize=figsize)
        x_pos = np.arange(len(names))

        ax.bar(
            x_pos,
            accuracies,
            yerr=[errors_lower, errors_upper],
            capsize=5,
            alpha=0.7,
            ecolor="black",
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy of  Jet reconstructors ({confidence*100:.0f}% CI)")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return fig, ax

    @staticmethod
    def _compute_binned_feature(
        binning_mask: np.ndarray, accuracy_data: np.ndarray, event_weights: np.ndarray
    ) -> np.ndarray:
        """
        Compute weighted accuracy in bins.

        Args:
            binning_mask: Boolean mask for binning (n_bins, n_events)
            accuracy_data: Binary accuracy data (n_events,)
            event_weights: Event weights (n_events,)

        Returns:
            Array of accuracies per bin
        """
        bin_weights = np.sum(event_weights.reshape(1, -1) * binning_mask, axis=1)
        bin_correct = np.sum(
            accuracy_data.reshape(1, -1) * event_weights.reshape(1, -1) * binning_mask,
            axis=1,
        )
        binned_accuracy = np.divide(
            bin_correct,
            bin_weights,
            out=np.zeros_like(bin_correct, dtype=float),
            where=bin_weights != 0,
        )
        return binned_accuracy

    def _bootstrap_binned_accuracy(
        self,
        reconstructor_index: int,
        binning_mask: np.ndarray,
        event_weights: np.ndarray,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute binned accuracy with bootstrap confidence intervals.

        Args:
            reconstructor_index: Index of the reconstructor in self.reconstructors
            binning_mask: Boolean mask for binning
            event_weights: Event weights
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level for intervals

        Returns:
            Tuple of (mean_accuracies, lower_bounds, upper_bounds) arrays
        """
        predictions = self.model_predictions[reconstructor_index]["assignment"]
        accuracy_data = self.evaluate_accuracy(
            self.y_test["assignment_labels"], predictions, per_event=True
        ).astype(float)
        return self._compute_binned_bootstrap(
            binning_mask, event_weights, n_bootstrap, confidence, accuracy_data
        )

    def _compute_binned_bootstrap(
        self, binning_mask, event_weights, n_bootstrap, confidence, accuracy_data
    ):
        n_samples = len(accuracy_data)
        n_bins = binning_mask.shape[0]
        bootstrap_binned_accuracies = np.zeros((n_bootstrap, n_bins))

        for i in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.randint(0, n_samples, size=n_samples)
            bootstrap_accuracy = accuracy_data[indices]
            bootstrap_weights = event_weights[indices]
            bootstrap_binning_mask = binning_mask[:, indices]

            binned_accuracy = self._compute_binned_feature(
                bootstrap_binning_mask, bootstrap_accuracy, bootstrap_weights
            )
            bootstrap_binned_accuracies[i] = binned_accuracy

        mean_accuracies = np.mean(bootstrap_binned_accuracies, axis=0)

        # Compute confidence intervals
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bounds = np.percentile(
            bootstrap_binned_accuracies, lower_percentile, axis=0
        )
        upper_bounds = np.percentile(
            bootstrap_binned_accuracies, upper_percentile, axis=0
        )

        return mean_accuracies, lower_bounds, upper_bounds

    def plot_binned_accuracy(
        self,
        feature_data_type: str,
        feature_name: str,
        fancy_feautre_label: Optional[str] = None,
        bins: int = 20,
        xlims: Optional[Tuple[float, float]] = None,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        show_errorbar: bool = True,
        show_combinatoric: bool = True,
    ):
        """
        Plot binned accuracy vs. a feature with bootstrap error bars.

        Args:
            feature_data_type: Type of feature ('jet', 'lepton', 'met')
            feature_name: Name of the feature
            bins: Number of bins or bin edges
            xlims: Optional x-axis limits
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level for error bars
            show_errorbar: Whether to show error bars
        """
        # Validate inputs
        if feature_data_type not in self.X_test:
            raise ValueError(
                f"Feature data type '{feature_data_type}' not found in test data."
            )
        if feature_name not in self.config.feature_indices[feature_data_type]:
            raise ValueError(
                f"Feature name '{feature_name}' not found in test data "
                f"for type '{feature_data_type}'."
            )

        # Extract feature data
        feature_idx = self.config.feature_indices[feature_data_type][feature_name]
        if self.X_test[feature_data_type].ndim == 2:
            feature_data = self.X_test[feature_data_type][:, feature_idx]
        elif self.X_test[feature_data_type].ndim == 3:
            feature_data = self.X_test[feature_data_type][:, feature_idx, 0]
        else:
            raise ValueError(
                f"Feature data for type '{feature_data_type}' has unsupported "
                f"number of dimensions: {self.X_test[feature_data_type].ndim}"
            )

        # Create figure if not provided
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create bins
        if xlims is not None:
            bins = np.linspace(xlims[0], xlims[1], bins + 1)
        else:
            bins = np.linspace(np.min(feature_data), np.max(feature_data), bins + 1)

        # Create binning mask
        binning_mask = (feature_data.reshape(1, -1) >= bins[:-1].reshape(-1, 1)) & (
            feature_data.reshape(1, -1) < bins[1:].reshape(-1, 1)
        )
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        if show_combinatoric:
            # Get binned number of jets
            num_jets = np.all(
                self.X_test["jet"] != self.config.padding_value, axis=-1
            ).sum(axis=-1)
            combinatoric_accuracy = 1 / (
                num_jets * (num_jets - 1)
            )  # Assuming 2 leptons to assign

            binned_combinatoric_accuracy = self._compute_binned_feature(
                binning_mask, combinatoric_accuracy, np.ones(feature_data.shape[0])
            )
            ax.plot(
                bin_centers,
                binned_combinatoric_accuracy,
                label="Random Assignment",
                color="black",
                linestyle="--",
            )

        # Get event weights
        event_weights = self.X_test.get("event_weight", np.ones(feature_data.shape[0]))

        # Compute binned accuracies with bootstrapping
        color_map = plt.get_cmap("tab10")

        print(f"\nComputing binned accuracy for {feature_name}...")
        for index, reconstructor in enumerate(self.reconstructors):
            if show_errorbar:
                mean_acc, lower, upper = self._bootstrap_binned_accuracy(
                    index, binning_mask, event_weights, n_bootstrap, confidence
                )
                errors_lower = mean_acc - lower
                errors_upper = upper - mean_acc
                ax.errorbar(
                    bin_centers,
                    mean_acc,
                    yerr=[errors_lower, errors_upper],
                    fmt="x",
                    label=reconstructor.get_name(),
                    color=color_map(index),
                    linestyle="None",
                )
            else:
                predictions = self.model_predictions[index]["assignment"]
                accuracy_data = self.evaluate_accuracy(
                    self.y_test["assignment_labels"], predictions, per_event=True
                ).astype(float)
                binned_accuracy = self._compute_binned_feature(
                    binning_mask, accuracy_data, event_weights
                )

                ax.plot(
                    bin_centers,
                    binned_accuracy,
                    label=reconstructor.get_name(),
                    color=color_map(index),
                )

        # Configure plot
        ax.set_xlabel(
            fancy_feautre_label if fancy_feautre_label is not None else feature_name
        )
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.set_xlim(bins[0], bins[-1])
        ax.grid(alpha=0.3)
        ax.legend(loc="best")

        # Add event count histogram
        ax_clone = ax.twinx()
        bin_counts = np.sum(
            event_weights.reshape(1, -1) * binning_mask, axis=1
        ) / np.sum(
            event_weights
        )  # Normalized counts
        ax_clone.bar(
            bin_centers,
            bin_counts,
            width=(bins[1] - bins[0]),
            alpha=0.2,
            color="red",
            label="Event Count",
        )
        ax_clone.set_ylabel("Event Count", color="red")
        ax_clone.tick_params(axis="y", labelcolor="red")

        title = f"Accuracy per Bin vs {fancy_feautre_label if fancy_feautre_label is not None else feature_name}"
        if show_errorbar:
            title += f" ({confidence*100:.0f}% CI)"
        ax.set_title(title)

        fig.tight_layout()
        return fig, ax

    def plot_confusion_matrices(
        self, normalize: bool = True, figsize_per_plot: Tuple[int, int] = (5, 5)
    ):
        """
        Plot confusion matrices for all reconstructors.

        Args:
            normalize: Whether to normalize the confusion matrix
            figsize_per_plot: Size of each subplot

        Returns:
            Tuple of (figure, axes)
        """
        number_reconstructors = len(self.reconstructors)
        rows = int(np.ceil(np.sqrt(number_reconstructors)))
        cols = int(np.ceil(number_reconstructors / rows))

        fig, axes = plt.subplots(
            rows, cols, figsize=(figsize_per_plot[0] * cols, figsize_per_plot[1] * rows)
        )
        axes = axes.flatten() if number_reconstructors > 1 else [axes]

        for i, reconstructor in enumerate(self.reconstructors):
            predictions = self.model_predictions[i]["assignment"]
            predicted_indices = np.argmax(predictions, axis=-2)
            true_indices = np.argmax(self.y_test["assignment_labels"], axis=-2)

            y_true = true_indices.flatten()
            y_pred = predicted_indices.flatten()

            cm = confusion_matrix(
                y_true, y_pred, normalize="true" if normalize else None
            )

            sns.heatmap(
                cm,
                annot=True,
                fmt=".2f" if normalize else "d",
                ax=axes[i],
                cmap="Blues",
                cbar_kws={"label": "Normalized Count" if normalize else "Count"},
            )
            axes[i].set_title(f"Confusion Matrix: {reconstructor.get_name()}")
            axes[i].set_xlabel("Predicted Label")
            axes[i].set_ylabel("True Label")

        # Remove unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        return fig, axes[: i + 1]

    def compute_complementarity_matrix(
        self,
    ) -> np.ndarray:
        """
        Compute complementarity matrix between reconstructors.

        Returns:
            Complementarity matrix (n_reconstructors, n_reconstructors)
        """
        n_reconstructors = len(self.reconstructors)
        complementarity_matrix = np.zeros((n_reconstructors, n_reconstructors))

        # Get per-event accuracy for each reconstructor
        per_event_accuracies = []
        for i in range(n_reconstructors):
            predictions = self.model_predictions[i]["assignment"]
            accuracy_data = self.evaluate_accuracy(
                self.y_test["assignment_labels"], predictions, per_event=True
            ).astype(float)
            per_event_accuracies.append(accuracy_data)

        # Compute complementarity
        for i in range(n_reconstructors):
            for j in range(n_reconstructors):
                if i != j:
                    both_incorrect = np.sum(
                        (1 - per_event_accuracies[i]) * (1 - per_event_accuracies[j])
                    )
                    total_events = len(per_event_accuracies[i])
                    complementarity_matrix[i, j] = both_incorrect / total_events
                else:
                    complementarity_matrix[i, j] = 0.0

        return complementarity_matrix

    def plot_complementarity_matrix(
        self,
        figsize: Tuple[int, int] = (8, 6),
    ):
        """
        Plot complementarity matrix between reconstructors.

        Args:
            figsize: Figure size

        Returns:
            Tuple of (figure, axis)
        """
        complementarity_matrix = self.compute_complementarity_matrix()
        names = [reconstructor.get_name() for reconstructor in self.reconstructors]

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            complementarity_matrix,
            annot=True,
            fmt=".2f",
            xticklabels=names,
            yticklabels=names,
            cmap="viridis",
            cbar_kws={"label": "Complementarity"},
            ax=ax,
        )
        ax.set_title("Complementarity Matrix between Reconstructors")
        ax.set_xlabel("")
        ax.set_ylabel("")

        fig.tight_layout()
        return fig, ax

    def _get_per_event_complementarity(
        self,
    ) -> np.ndarray:
        """
        Get per-event success for at least one reconstructor.
        Returns:
            Array of per-event complementarity (n_events,)
        """
        n_reconstructors = len(self.reconstructors)
        n_events = self.X_test[list(self.X_test.keys())[0]].shape[0]
        per_event_success = np.zeros((n_events, n_reconstructors), dtype=bool)

        # Get per-event accuracy for each reconstructor
        for i in range(n_reconstructors):
            predictions = self.model_predictions[i]["assignment"]
            accuracy_data = self.evaluate_accuracy(
                self.y_test["assignment_labels"], predictions, per_event=True
            ).astype(bool)
            per_event_success[:, i] = accuracy_data

        # Compute per-event complementarity
        per_event_complementarity = np.any(per_event_success, axis=1).astype(float)

        return per_event_complementarity

    def plot_conditional_complementarity_matrix(
        self,
        figsize: Tuple[int, int] = (8, 6),
    ):
        # Todo: Implement this method
        pass

    def plot_binned_complementarity(
        self,
        feature_data_type: str,
        feature_name: str,
        fancy_feautre_label: Optional[str] = None,
        bins: int = 20,
        xlims: Optional[Tuple[float, float]] = None,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        show_errorbar: bool = True,
    ):
        """
        Plot binned complementarity vs. a feature with bootstrap error bars.

        Args:
            feature_data_type: Type of feature ('jet', 'lepton', 'met')
            feature_name: Name of the feature
            bins: Number of bins or bin edges
            xlims: Optional x-axis limits
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level for error bars
            show_errorbar: Whether to show error bars
        """
        # Validate inputs
        if feature_data_type not in self.X_test:
            raise ValueError(
                f"Feature data type '{feature_data_type}' not found in test data."
            )
        if feature_name not in self.config.feature_indices[feature_data_type]:
            raise ValueError(
                f"Feature name '{feature_name}' not found in test data "
                f"for type '{feature_data_type}'."
            )

        # Extract feature data
        feature_idx = self.config.feature_indices[feature_data_type][feature_name]
        if self.X_test[feature_data_type].ndim == 2:
            feature_data = self.X_test[feature_data_type][:, feature_idx]
        elif self.X_test[feature_data_type].ndim == 3:
            feature_data = self.X_test[feature_data_type][:, feature_idx, 0]
        else:
            raise ValueError(
                f"Feature data for type '{feature_data_type}' has unsupported "
                f"number of dimensions: {self.X_test[feature_data_type].ndim}"
            )

        # Create figure if not provided
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create bins
        if xlims is not None:
            bins = np.linspace(xlims[0], xlims[1], bins + 1)
        else:
            bins = np.linspace(np.min(feature_data), np.max(feature_data), bins + 1)

        # Create binning mask
        binning_mask = (feature_data.reshape(1, -1) >= bins[:-1].reshape(-1, 1)) & (
            feature_data.reshape(1, -1) < bins[1:].reshape(-1, 1)
        )
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        # Get per-event complementarity
        per_event_complementarity = self._get_per_event_complementarity()
        # Get event weights
        event_weights = self.X_test.get("event_weight", np.ones(feature_data.shape[0]))
        # Compute binned complementarity with bootstrapping
        print(f"\nComputing binned complementarity for {feature_name}...")
        if show_errorbar:
            mean_comp, lower, upper = self._compute_binned_bootstrap(
                binning_mask,
                event_weights,
                n_bootstrap,
                confidence,
                per_event_complementarity,
            )
            errors_lower = mean_comp - lower
            errors_upper = upper - mean_comp
            ax.errorbar(
                bin_centers,
                mean_comp,
                yerr=[errors_lower, errors_upper],
                fmt="x",
                label="Complementarity",
                color="blue",
                linestyle="None",
            )
        else:
            binned_complementarity = self._compute_binned_feature(
                binning_mask, per_event_complementarity, event_weights
            )
            ax.plot(
                bin_centers,
                binned_complementarity,
                label="Complementarity",
                color="blue",
            )
        # Add event count histogram
        ax_clone = ax.twinx()
        bin_counts = np.sum(
            event_weights.reshape(1, -1) * binning_mask, axis=1
        ) / np.sum(
            event_weights
        )  # Normalized counts
        ax_clone.bar(
            bin_centers,
            bin_counts,
            width=(bins[1] - bins[0]),
            alpha=0.2,
            color="red",
            label="Event Count",
        )
        ax_clone.set_ylabel("Event Count", color="red")
        ax_clone.tick_params(axis="y", labelcolor="red")

        # Configure plot
        ax.set_xlabel(
            fancy_feautre_label if fancy_feautre_label is not None else feature_name
        )
        ax.set_ylabel("Complementarity")
        ax.set_ylim(0, 1)
        ax.set_xlim(bins[0], bins[-1])
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        ax.set_title(
            f"Complementarity per Bin vs {fancy_feautre_label if fancy_feautre_label is not None else feature_name}"
            + (f" ({confidence*100:.0f}% CI)" if show_errorbar else "")
        )
        fig.tight_layout()
        return fig, ax

    def _compute_top_lorentz_vectors(
        self, reconstructor_index: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute top reconstructions for each event.

        Args:
            reconstructor_index: Index of the reconstructor in self.reconstructors
        Returns:
            Tuple of (top_indices, top_probabilities)
        """
        assignment_predictions = self.model_predictions[reconstructor_index][
            "assignment"
        ]
        neutrino_predictions = self.model_predictions[reconstructor_index]["regression"]
        lepton_features = self.X_test["lepton"][:, :, :4]
        jet_features = self.X_test["jet"][:, :, :4]
        selected_jet_indices = assignment_predictions.argmax(axis=-2)

        reco_jets = np.take_along_axis(
            jet_features,
            selected_jet_indices[:, :, np.newaxis],
            axis=1,
        )
        reco_leptons = lepton_features.reshape(-1, 2, 4)
        reco_neutrinos = neutrino_predictions.reshape(-1, 2, 3)

        reco_jets_p4 = lorentz_vector_from_PtEtaPhiE_array(reco_jets)
        reco_leptons_p4 = lorentz_vector_from_PtEtaPhiE_array(reco_leptons)
        reco_neutrinos_p4 = lorentz_vector_from_neutrino_momenta_array(reco_neutrinos)
        top1_p4 = reco_jets_p4[:, 0] + reco_leptons_p4[:, 0] + reco_neutrinos_p4[:, 0]
        top2_p4 = reco_jets_p4[:, 1] + reco_leptons_p4[:, 1] + reco_neutrinos_p4[:, 1]
        return top1_p4, top2_p4

    def plot_binned_top_mass_resolution(
        self,
        feature_data_type: str,
        feature_name: str,
        true_top_mass_labels: List[str] = ["truth_top_mass", "truth_tbar_mass"],
        fancy_feautre_label: Optional[str] = None,
        bins: int = 20,
        xlims: Optional[Tuple[float, float]] = None,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        show_errorbar: bool = True,
    ):
        """
        Plot binned resolution of the reconstructed top vs. a feature with bootstrap error bars.
        Args:
            feature_data_type: Type of feature ('jet', 'lepton', 'met')
            feature_name: Name of the feature
            bins: Number of bins or bin edges
            xlims: Optional x-axis limits
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level for error bars
            show_errorbar: Whether to show error bars
            show_combinatoric: Whether to show combinatoric baseline
        Returns:
            Tuple of (figure, axis)
        """
        # Validate inputs
        if feature_data_type not in self.X_test:
            raise ValueError(
                f"Feature data type '{feature_data_type}' not found in test data."
            )
        if feature_name not in self.config.feature_indices[feature_data_type]:
            raise ValueError(
                f"Feature name '{feature_name}' not found in test data "
                f"for type '{feature_data_type}'."
            )
        if any(
            [
                label not in self.config.feature_indices["non_training"]
                for label in true_top_mass_labels
            ]
        ):
            raise ValueError(
                f"One or more true top mass labels not found in test data "
                f"for type 'non_training'."
            )
        else:
            true_top1_mass = self.X_test["non_training"][
                :, self.config.feature_indices["non_training"][true_top_mass_labels[0]]
            ]
            true_top2_mass = self.X_test["non_training"][
                :, self.config.feature_indices["non_training"][true_top_mass_labels[1]]
            ]

        # Extract feature data
        feature_idx = self.config.feature_indices[feature_data_type][feature_name]
        if self.X_test[feature_data_type].ndim == 2:
            feature_data = self.X_test[feature_data_type][:, feature_idx]
        elif self.X_test[feature_data_type].ndim == 3:
            feature_data = self.X_test[feature_data_type][:, feature_idx, 0]
        else:
            raise ValueError(
                f"Feature data for type '{feature_data_type}' has unsupported "
                f"number of dimensions: {self.X_test[feature_data_type].ndim}"
            )

        # Create figure if not provided
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create bins
        if xlims is not None:
            bins = np.linspace(xlims[0], xlims[1], bins + 1)
        else:
            bins = np.linspace(np.min(feature_data), np.max(feature_data), bins + 1)

        # Create binning mask
        binning_mask = (feature_data.reshape(1, -1) >= bins[:-1].reshape(-1, 1)) & (
            feature_data.reshape(1, -1) < bins[1:].reshape(-1, 1)
        )
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # Get event weights
        event_weights = self.X_test.get("event_weight", np.ones(feature_data.shape[0]))

        # Compute top mass resolutions with bootstrapping
        color_map = plt.get_cmap("tab10")

        print(f"\nComputing binned top mass resolution for {feature_name}...")
        for index, reconstructor in enumerate(self.reconstructors):
            # Compute top masses
            top1_p4, top2_p4 = self._compute_top_lorentz_vectors(index)
            top1_mass = compute_mass_from_lorentz_vector_array(top1_p4)
            top2_mass = compute_mass_from_lorentz_vector_array(top2_p4)

            mass_deviation_extended = np.concatenate(
                [
                    np.abs(top1_mass - true_top1_mass) / true_top1_mass,
                    np.abs(top2_mass - true_top2_mass) / true_top2_mass,
                ],
                axis=0,
            )
            binning_mask_extended = np.concatenate(
                [binning_mask, binning_mask], axis=1
            )  # Extend for both tops
            event_weights_extended = np.concatenate(
                [event_weights, event_weights], axis=0
            )  # Extend for both tops

            binned_mean_mass_deviation = (np.sum(
                event_weights_extended.reshape(1, -1)
                * mass_deviation_extended.reshape(1, -1)
                * binning_mask_extended,
                axis=1,
            ) / np.sum(
                event_weights_extended.reshape(1, -1) * binning_mask_extended, axis=1
            ))
            binned_mass_squared_deviation = (np.sum(
                event_weights_extended.reshape(1, -1)
                * (mass_deviation_extended**2).reshape(1, -1)
                * binning_mask_extended,
                axis=1,
            ) / np.sum(
                event_weights_extended.reshape(1, -1) * binning_mask_extended, axis=1
            ))
            binned_resolution = np.sqrt(
                binned_mass_squared_deviation - binned_mean_mass_deviation**2
            )

            
            if show_errorbar:
                # Bootstrap for error bars
                (
                    mean_res,
                    lower,
                    upper,
                ) = self._compute_binned_bootstrap(
                    binning_mask_extended,
                    event_weights_extended,
                    n_bootstrap,
                    confidence,
                    mass_deviation_extended,
                )
                errors_lower = mean_res - lower
                errors_upper = upper - mean_res
                ax.errorbar(
                    bin_centers,
                    mean_res,
                    yerr=[errors_lower, errors_upper],
                    fmt="x",
                    label=reconstructor.get_name(),
                    color=color_map(index),
                    linestyle="None",
                )
            else:
                ax.plot(
                    bin_centers,
                    binned_resolution,
                    label=reconstructor.get_name(),
                    color=color_map(index),
                )
        # Configure plot
        ax.set_xlabel(
            fancy_feautre_label if fancy_feautre_label is not None else feature_name
        )
        ax.set_ylabel("Top Mass Resolution")
        ax.set_ylim(0, None)
        ax.set_xlim(bins[0], bins[-1])
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        ax.set_title(
            f"Top Mass Resolution per Bin vs {fancy_feautre_label if fancy_feautre_label is not None else feature_name}"
            + (f" ({confidence*100:.0f}% CI)" if show_errorbar else "")
        )
        # Add event count histogram
        ax_clone = ax.twinx()
        bin_counts = np.sum(
            event_weights.reshape(1, -1) * binning_mask, axis=1
        ) / np.sum(
            event_weights
        )  # Normalized counts
        ax_clone.bar(
            bin_centers,
            bin_counts,
            width=(bins[1] - bins[0]),
            alpha=0.2,
            color="red",
            label="Event Count",
        )
        ax_clone.set_ylabel("Event Count", color="red")
        ax_clone.tick_params(axis="y", labelcolor="red")
        fig.tight_layout()
        return fig, ax




    def plot_binned_ttbar_mass_resolution(
        self,
        feature_data_type: str,
        feature_name: str,
        true_ttbar_mass_label: str = [ "truth_ttbar_mass"],
        fancy_feautre_label: Optional[str] = None,
        bins: int = 20,
        xlims: Optional[Tuple[float, float]] = None,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        show_errorbar: bool = True,
    ):
        """
        Plot binned resolution of the reconstructed top vs. a feature with bootstrap error bars.
        Args:
            feature_data_type: Type of feature ('jet', 'lepton', 'met')
            feature_name: Name of the feature
            bins: Number of bins or bin edges
            xlims: Optional x-axis limits
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level for error bars
            show_errorbar: Whether to show error bars
            show_combinatoric: Whether to show combinatoric baseline
        Returns:
            Tuple of (figure, axis)
        """
        # Validate inputs
        if feature_data_type not in self.X_test:
            raise ValueError(
                f"Feature data type '{feature_data_type}' not found in test data."
            )
        if feature_name not in self.config.feature_indices[feature_data_type]:
            raise ValueError(
                f"Feature name '{feature_name}' not found in test data "
                f"for type '{feature_data_type}'."
            )
        if any(
            [
                label not in self.config.feature_indices["non_training"]
                for label in true_ttbar_mass_label
            ]
        ):
            raise ValueError(
                f"One or more true top mass labels not found in test data "
                f"for type 'non_training'."
            )
        else:
            true_ttbar_mass = self.X_test["non_training"][
                :, self.config.feature_indices["non_training"][true_ttbar_mass_label[0]]
            ]

        # Extract feature data
        feature_idx = self.config.feature_indices[feature_data_type][feature_name]
        if self.X_test[feature_data_type].ndim == 2:
            feature_data = self.X_test[feature_data_type][:, feature_idx]
        elif self.X_test[feature_data_type].ndim == 3:
            feature_data = self.X_test[feature_data_type][:, feature_idx, 0]
        else:
            raise ValueError(
                f"Feature data for type '{feature_data_type}' has unsupported "
                f"number of dimensions: {self.X_test[feature_data_type].ndim}"
            )

        # Create figure if not provided
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create bins
        if xlims is not None:
            bins = np.linspace(xlims[0], xlims[1], bins + 1)
        else:
            bins = np.linspace(np.min(feature_data), np.max(feature_data), bins + 1)

        # Create binning mask
        binning_mask = (feature_data.reshape(1, -1) >= bins[:-1].reshape(-1, 1)) & (
            feature_data.reshape(1, -1) < bins[1:].reshape(-1, 1)
        )
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # Get event weights
        event_weights = self.X_test.get("event_weight", np.ones(feature_data.shape[0]))

        # Compute top mass resolutions with bootstrapping
        color_map = plt.get_cmap("tab10")

        print(f"\nComputing binned top mass resolution for {feature_name}...")
        for index, reconstructor in enumerate(self.reconstructors):
            # Compute top masses
            top1_p4, top2_p4 = self._compute_top_lorentz_vectors(index)
            ttbar_p4 = top1_p4 + top2_p4
            ttbar_mass = compute_mass_from_lorentz_vector_array(ttbar_p4)

            mass_deviation = np.abs(ttbar_mass - true_ttbar_mass) / true_ttbar_mass

            binned_mean_mass_deviation = (np.sum(
                event_weights.reshape(1, -1)
                * mass_deviation.reshape(1, -1)
                * binning_mask,
                axis=1,
            ) / np.sum(
                event_weights.reshape(1, -1) * binning_mask, axis=1
            ))
            binned_mass_squared_deviation = (np.sum(
                event_weights.reshape(1, -1)
                * (mass_deviation**2).reshape(1, -1)
                * binning_mask,
                axis=1,
            ) / np.sum(
                event_weights.reshape(1, -1) * binning_mask, axis=1
            ))
            binned_resolution = np.sqrt(
                binned_mass_squared_deviation - binned_mean_mass_deviation**2
            )

            if show_errorbar:
                # Bootstrap for error bars
                (
                    mean_res,
                    lower,
                    upper,
                ) = self._compute_binned_bootstrap(
                    binning_mask,
                    event_weights,
                    n_bootstrap,
                    confidence,
                    mass_deviation,
                )
                errors_lower = mean_res - lower
                errors_upper = upper - mean_res
                ax.errorbar(
                    bin_centers,
                    mean_res,
                    yerr=[errors_lower, errors_upper],
                    fmt="x",
                    label=reconstructor.get_name(),
                    color=color_map(index),
                    linestyle="None",
                )
            else:
                ax.plot(
                    bin_centers,
                    binned_resolution,
                    label=reconstructor.get_name(),
                    color=color_map(index),
                )
        # Configure plot
        ax.set_xlabel(
            fancy_feautre_label if fancy_feautre_label is not None else feature_name
        )
        ax.set_ylabel("Top Mass Resolution")
        ax.set_ylim(0, None)
        ax.set_xlim(bins[0], bins[-1])
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
        ax.set_title(
            f"Top Mass Resolution per Bin vs {fancy_feautre_label if fancy_feautre_label is not None else feature_name}"
            + (f" ({confidence*100:.0f}% CI)" if show_errorbar else "")
        )
        # Add event count histogram
        ax_clone = ax.twinx()
        bin_counts = np.sum(
            event_weights.reshape(1, -1) * binning_mask, axis=1
        ) / np.sum(
            event_weights
        )  # Normalized counts
        ax_clone.bar(
            bin_centers,
            bin_counts,
            width=(bins[1] - bins[0]),
            alpha=0.2,
            color="red",
            label="Event Count",
        )
        ax_clone.set_ylabel("Event Count", color="red")
        ax_clone.tick_params(axis="y", labelcolor="red")
        fig.tight_layout()
        return fig, ax
