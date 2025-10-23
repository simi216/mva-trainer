import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from typing import Union, Optional, Tuple
from copy import deepcopy

from .Assignment import JetAssignerBase, MLAssignerBase


class MLEvaluatorBase:
    """Base evaluator for ML-based jet assignment models."""
    
    def __init__(self, assigner: MLAssignerBase, X_test, y_test):
        self.assigner = assigner
        self.X_test = X_test
        self.y_test = y_test
        self.max_leptons = assigner.max_leptons
        self.max_jets = assigner.max_jets
        self.met_features = assigner.met_features
        self.n_jets: int = assigner.n_jets
        self.n_leptons: int = assigner.n_leptons
        self.n_met: int = assigner.n_met
        self.padding_value: float = assigner.padding_value
        self.feature_index_dict = assigner.feature_index_dict

    def evaluate(self):
        """Evaluate the model on test data."""
        if self.X_test is None or self.y_test is None:
            raise ValueError(
                "Test data not loaded. Please load test data before evaluation."
            )

        results = self.model.evaluate(self.X_test, self.y_test, verbose=2)
        print("Test Loss and Metrics:", results)
        return results

    def plot_training_history(self):
        """Plot training and validation loss/accuracy over epochs."""
        if self.assigner.history is None:
            raise ValueError(
                "No training history found. Please train the model before plotting history."
            )

        history = self.assigner.history

        # Plot loss
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(history.history['loss'], label='Training Loss')
        ax[0].plot(history.history['val_loss'], label='Validation Loss')
        ax[0].set_title('Model Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()
        # Plot accuracy
        ax[1].plot(history.history['assignment_accuracy'], label='Training Accuracy')
        ax[1].plot(history.history['val_assignment_accuracy'], label='Validation Accuracy')
        ax[1].set_title('Model Accuracy')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()
        return fig, ax
        

    def get_assigner_name(self) -> str:
        """Get the name of the assigner."""
        return self.assigner.get_name()

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

        baseline_performance = self.assigner.evaluate_accuracy(self.X_test, self.y_test)
        print(f"Baseline Performance: {baseline_performance:.4f}")

        importances = {}
        
        # Jet features
        for feature, feature_idx in self.feature_index_dict["jet"].items():
            scores = []
            for _ in range(num_repeats):
                X_permuted = deepcopy(self.X_test)
                mask = np.any(X_permuted["jet"] != self.padding_value, axis = -1)
                X_permuted["jet"][mask, feature_idx] = np.random.permutation(
                    X_permuted["jet"][mask, feature_idx]
                )
                permuted_performance = self.assigner.evaluate_accuracy(
                    X_permuted, self.y_test
                )
                scores.append(baseline_performance - permuted_performance)
            importances[feature] = np.mean(scores)

        # Lepton features
        for feature, feature_idx in self.feature_index_dict["lepton"].items():
            scores = []
            for _ in range(num_repeats):
                X_permuted = deepcopy(self.X_test)
                X_permuted["lepton"][:, :, feature_idx] = np.random.permutation(
                    X_permuted["lepton"][:, :, feature_idx]
                )
                permuted_performance = self.assigner.evaluate_accuracy(
                    X_permuted, self.y_test
                )
                scores.append(baseline_performance - permuted_performance)
            importances[feature] = np.mean(scores)

        # met features
        if self.met_features:
            for feature, feature_idx in self.feature_index_dict["met"].items():
                scores = []
                for _ in range(num_repeats):
                    X_permuted = deepcopy(self.X_test)
                    X_permuted["met"][:, :, feature_idx] = np.random.permutation(
                        X_permuted["met"][:, :, feature_idx]
                    )
                    permuted_performance = self.assigner.evaluate_accuracy(
                        X_permuted, self.y_test
                    )
                    scores.append(baseline_performance - permuted_performance)
                importances[feature] = np.mean(scores)
                
        return importances

    def plot_feature_importance(self, num_repeats = 10):

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


class JetAssignmentEvaluator:
    """Evaluator for comparing multiple jet assignment algorithms."""
    
    def __init__(
        self, 
        assigners: Union[list[JetAssignerBase], JetAssignerBase], 
        X_test, 
        y_test
    ):
        if isinstance(assigners, JetAssignerBase):
            self.assigners = [assigners]
        else:
            self.assigners = assigners

        self.X_test = X_test
        self.y_test = np.argmax(y_test, axis=-2)
        
        # Validate that all assigners have the same configuration
        configs = [assigner.config for assigner in self.assigners]
        config_1 = configs[0]
        for config in configs[1:]:
            if config != config_1:
                raise ValueError(
                    "All assigners must have the same DataConfig for consistent evaluation."
                )

        self.config = config_1
        self.feature_index_dict = config_1.get_feature_index_dict()

    def evaluate_all(self) -> dict:
        """
        Evaluate accuracy for all assigners.
        
        Returns:
            Dictionary mapping assigner names to accuracy scores
        """
        results = {}
        for assigner in self.assigners:
            predictions = assigner.predict_indices(self.X_test)
            predicted_indices = np.argmax(predictions, axis=-2)
            accuracy = np.mean(predicted_indices == self.y_test)
            results[assigner.get_name()] = accuracy
            print(f"Accuracy for {assigner.get_name()}: {accuracy:.4f}")
        return results

    def _bootstrap_accuracy(
        self, 
        assigner: JetAssignerBase, 
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Compute accuracy with bootstrap confidence intervals.
        
        Args:
            assigner: Jet assigner to evaluate
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level for intervals
            
        Returns:
            Tuple of (mean_accuracy, lower_bound, upper_bound)
        """
        predictions = assigner.predict_indices(self.X_test)
        predicted_indices = np.argmax(predictions, axis=-2)
        
        n_samples = len(self.y_test)
        bootstrap_accuracies = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.randint(0, n_samples, size=n_samples)
            bootstrap_predictions = predicted_indices[indices]
            bootstrap_y_test = self.y_test[indices]
            
            accuracy = np.mean(bootstrap_predictions == bootstrap_y_test)
            bootstrap_accuracies.append(accuracy)
        
        bootstrap_accuracies = np.array(bootstrap_accuracies)
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
        figsize: Tuple[int, int] = (10, 6)
    ):
        """
        Plot accuracies for all assigners with error bars from bootstrapping.
        
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
        for assigner in self.assigners:
            mean_acc, lower, upper = self._bootstrap_accuracy(
                assigner, n_bootstrap, confidence
            )
            names.append(assigner.get_name())
            accuracies.append(mean_acc)
            errors_lower.append(mean_acc - lower)
            errors_upper.append(upper - mean_acc)
            
            print(f"{assigner.get_name()}: {mean_acc:.4f} "
                  f"[{lower:.4f}, {upper:.4f}]")
        
        fig, ax = plt.subplots(figsize=figsize)
        x_pos = np.arange(len(names))
        
        ax.bar(x_pos, accuracies, yerr=[errors_lower, errors_upper], 
               capsize=5, alpha=0.7, ecolor='black')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Accuracy of Different Jet Assigners ({confidence*100:.0f}% CI)")
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _compute_binned_accuracy(
        binning_mask: np.ndarray, 
        accuracy_data: np.ndarray, 
        event_weights: np.ndarray
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
        assigner: JetAssignerBase,
        binning_mask: np.ndarray,
        event_weights: np.ndarray,
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute binned accuracy with bootstrap confidence intervals.
        
        Args:
            assigner: Jet assigner to evaluate
            binning_mask: Boolean mask for binning
            event_weights: Event weights
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level for intervals
            
        Returns:
            Tuple of (mean_accuracies, lower_bounds, upper_bounds) arrays
        """
        predictions = assigner.predict_indices(self.X_test)
        predicted_indices = np.argmax(predictions, axis=-2)
        accuracy_data = np.all(predicted_indices == self.y_test, axis=-1).astype(float)
        
        n_samples = len(accuracy_data)
        n_bins = binning_mask.shape[0]
        bootstrap_accuracies = np.zeros((n_bootstrap, n_bins))
        
        for i in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.randint(0, n_samples, size=n_samples)
            bootstrap_accuracy = accuracy_data[indices]
            bootstrap_weights = event_weights[indices]
            bootstrap_mask = binning_mask[:, indices]
            
            binned_acc = self._compute_binned_accuracy(
                bootstrap_mask, bootstrap_accuracy, bootstrap_weights
            )
            bootstrap_accuracies[i] = binned_acc
        
        mean_accuracies = np.nanmean(bootstrap_accuracies, axis=0)
        
        # Compute confidence intervals
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bounds = np.nanpercentile(bootstrap_accuracies, lower_percentile, axis=0)
        upper_bounds = np.nanpercentile(bootstrap_accuracies, upper_percentile, axis=0)
        
        return mean_accuracies, lower_bounds, upper_bounds

    def plot_binned_accuracy(
        self,
        feature_data_type: str,
        feature_name: str,
        bins: int = 20,
        xlims: Optional[Tuple[float, float]] = None,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        color_map=None,
        label: Optional[str] = None,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        show_errorbar: bool = True
    ):
        """
        Plot binned accuracy vs. a feature with bootstrap error bars.
        
        Args:
            feature_data_type: Type of feature ('jet', 'lepton', 'met')
            feature_name: Name of the feature
            bins: Number of bins or bin edges
            xlims: Optional x-axis limits
            fig: Optional existing figure
            ax: Optional existing axes
            color_map: Optional color map for plotting
            label: Optional legend label
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level for error bars
            show_errorbar: Whether to show error bars
        """
        # Validate inputs
        if feature_data_type not in self.X_test:
            raise ValueError(
                f"Feature data type '{feature_data_type}' not found in test data."
            )
        if feature_name not in self.feature_index_dict[feature_data_type]:
            raise ValueError(
                f"Feature name '{feature_name}' not found in test data "
                f"for type '{feature_data_type}'."
            )
        
        # Extract feature data
        feature_idx = self.feature_index_dict[feature_data_type][feature_name]
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
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bins
        if xlims is not None:
            bins = np.linspace(xlims[0], xlims[1], bins + 1)
        else:
            bins = np.linspace(np.min(feature_data), np.max(feature_data), bins + 1)
        
        # Create binning mask
        binning_mask = (
            (feature_data.reshape(1, -1) >= bins[:-1].reshape(-1, 1)) & 
            (feature_data.reshape(1, -1) < bins[1:].reshape(-1, 1))
        )
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        # Get event weights
        event_weights = self.X_test.get("event_weight", np.ones(feature_data.shape[0]))
        
        # Compute binned accuracies with bootstrapping
        color_map = plt.get_cmap('tab10') if color_map is None else color_map
        
        print(f"\nComputing binned accuracy for {feature_name}...")
        for index, assigner in enumerate(self.assigners):
            if show_errorbar:
                mean_acc, lower, upper = self._bootstrap_binned_accuracy(
                    assigner, binning_mask, event_weights, n_bootstrap, confidence
                )
                errors_lower = mean_acc - lower
                errors_upper = upper - mean_acc
                
                ax.errorbar(
                    bin_centers,
                    mean_acc,
                    yerr=[errors_lower, errors_upper],
                    fmt='x',
                    label=assigner.get_name(),
                    color=color_map(index),
                    linestyle='None',
                )
            else:
                predictions = assigner.predict_indices(self.X_test)
                predicted_indices = np.argmax(predictions, axis=-2)
                accuracy_data = np.all(
                    predicted_indices == self.y_test, axis=-1
                ).astype(float)
                binned_accuracy = self._compute_binned_accuracy(
                    binning_mask, accuracy_data, event_weights
                )
                
                ax.plot(
                    bin_centers,
                    binned_accuracy,
                    marker='o',
                    label=assigner.get_name(),
                    color=color_map(index),
                    linewidth=1.5,
                    markersize=6,
                    capsize=3,
                )
        
        # Configure plot
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.set_xlim(bins[0], bins[-1])
        ax.grid(alpha=0.3)
        
        if label is not None:
            ax.legend(title=label)
        else:
            ax.legend()
        
        # Add event count histogram
        ax_clone = ax.twinx()
        bin_counts = np.sum(event_weights.reshape(1, -1) * binning_mask, axis=1)
        ax_clone.bar(
            bin_centers,
            bin_counts,
            width=(bins[1] - bins[0]),
            alpha=0.2,
            color="gray",
            label="Event Count",
        )
        ax_clone.set_ylabel("Event Count")
        
        title = f"Binned Accuracy vs {feature_name}"
        if show_errorbar:
            title += f" ({confidence*100:.0f}% CI)"
        plt.title(title)
        
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrices(
        self, 
        normalize: bool = True,
        figsize_per_plot: Tuple[int, int] = (5, 5)
    ):
        """
        Plot confusion matrices for all assigners.
        
        Args:
            normalize: Whether to normalize the confusion matrix
            figsize_per_plot: Size of each subplot
            
        Returns:
            Tuple of (figure, axes)
        """
        number_assigners = len(self.assigners)
        rows = int(np.ceil(np.sqrt(number_assigners)))
        cols = int(np.ceil(number_assigners / rows))
        
        fig, axes = plt.subplots(
            rows, cols, 
            figsize=(figsize_per_plot[0] * cols, figsize_per_plot[1] * rows)
        )
        axes = axes.flatten() if number_assigners > 1 else [axes]
        
        for i, assigner in enumerate(self.assigners):
            predictions = assigner.predict_indices(self.X_test)
            predicted_indices = np.argmax(predictions, axis=-2)
            
            y_true = self.y_test.flatten()
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
                cbar_kws={'label': 'Normalized Count' if normalize else 'Count'}
            )
            axes[i].set_title(f"Confusion Matrix: {assigner.get_name()}")
            axes[i].set_xlabel("Predicted Label")
            axes[i].set_ylabel("True Label")
        
        # Remove unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()
        
        return fig, axes[:i + 1]