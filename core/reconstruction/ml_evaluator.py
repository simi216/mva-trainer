"""Evaluator for ML-based jet assignment models."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from copy import deepcopy

from .evaluator_base import BootstrapCalculator, AccuracyCalculator, NeutrinoDeviationCalculator
from . import MLReconstructorBase


class FeatureImportanceCalculator:
    """Calculate feature importance using permutation importance."""

    def __init__(
        self,
        reconstructor: MLReconstructorBase,
        X_test: dict,
        y_test: dict,
    ):
        self.reconstructor = reconstructor
        self.X_test = X_test
        self.y_test = y_test
        self.config = reconstructor.config
        self.padding_value = reconstructor.padding_value

    def compute_permutation_importance(
        self, num_repeats: int = 5, evaluate_regression=False
    ) -> dict:
        """
        Compute feature importance using permutation importance.

        Args:
            num_repeats: Number of times to permute each feature

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.X_test is None or self.y_test is None:
            raise ValueError(
                "Test data not loaded. Please load test data before computing "
                "permutation importance."
            )

        assignment_baseline_prediction, regression_baseline_prediction = (
            self.reconstructor.complete_forward_pass(self.X_test)
        )

        assignment_baseline_performance = AccuracyCalculator.compute_accuracy(
            assignment_baseline_prediction,
            self.y_test["assignment_labels"],
        )
        regression_baseline_prediction = None
        print(f"Baseline Assignment Performance: {assignment_baseline_performance:.4f}")

        if evaluate_regression:
            regression_baseline_performance = (
                NeutrinoDeviationCalculator.compute_relative_deviation(
                    regression_baseline_prediction,
                    self.y_test["nu_flows_regression_targets"],
                )
            )
            print(
                f"Baseline Regression Performance: {regression_baseline_performance:.4f}"
            )

        assingment_importances = {}
        regression_importances = {}

        # Compute importance for jet features

        assingment_jet_importances, regression_jet_importances = (
            self._compute_feature_type_importance(
                "jet",
                num_repeats,
                assignment_baseline_performance,
                regression_baseline_performance,
            )
        )

        assingment_importances.update(assingment_jet_importances)
        if evaluate_regression:
            regression_importances.update(regression_jet_importances)

        # Compute importance for lepton features
        assignment_lepton_importances, regression_lepton_importances = (
            self._compute_feature_type_importance(
                "lepton",
                num_repeats,
                assignment_baseline_performance,
                regression_baseline_performance,
            )
        )

        assingment_importances.update(assignment_lepton_importances)
        if evaluate_regression:
            regression_importances.update(regression_lepton_importances)

        # Compute importance for MET features if available
        if "met" in self.X_test:
            assignment_met_importances, regression_met_importances = (
                self._compute_feature_type_importance(
                    "met",
                    num_repeats,
                    assignment_baseline_performance,
                    regression_baseline_performance,
                )
            )

            assingment_importances.update(assignment_met_importances)
            if evaluate_regression:
                regression_importances.update(regression_met_importances)

        return (
            assingment_importances
            if not evaluate_regression
            else (assingment_importances, regression_importances)
        )

    def _compute_feature_type_importance(
        self,
        feature_type: str,
        num_repeats: int,
        assignment_baseline_performance: float,
        regression_baseline_performance: Optional[float] = None,
    ) -> dict:
        """Compute importance for a specific feature type."""
        assignment_importances = {}
        regression_importances = {}
        feature_indices = self.config.feature_indices[feature_type]

        for feature_name, feature_idx in feature_indices.items():
            assignment_scores = []
            regression_scores = []

            for _ in range(num_repeats):
                X_permuted = deepcopy(self.X_test)

                if feature_type == "jet":
                    # Only permute non-padded jet features
                    mask = np.any(X_permuted["jet"] != self.padding_value, axis=-1)
                    X_permuted["jet"][mask, feature_idx] = np.random.permutation(
                        X_permuted["jet"][mask, feature_idx]
                    )
                else:
                    # Permute lepton/MET features directly
                    X_permuted[feature_type][:, :, feature_idx] = np.random.permutation(
                        X_permuted[feature_type][:, :, feature_idx]
                    )

                permutated_assignment_pred, permutated_regression_pred = (
                    self.reconstructor.complete_forward_pass(X_permuted)
                )
                assignment_performance = (
                    AccuracyCalculator.compute_accuracy(
                        permutated_assignment_pred, self.y_test["assignment_labels"]
                    )
                    - assignment_baseline_performance
                )
                assignment_scores.append(assignment_performance)
                if regression_baseline_performance is not None:
                    regression_performance = (
                        NeutrinoDeviationCalculator.compute_relative_deviation(
                            permutated_regression_pred,
                            self.y_test["nu_flows_regression_targets"],
                        )
                        - regression_baseline_performance
                    )
                    regression_scores.append(regression_performance)

        return assignment_importances, regression_importances


class MLEvaluator:
    """Evaluator for ML-based jet assignment models."""

    def __init__(
        self,
        reconstructor: MLReconstructorBase,
        X_test: dict,
        y_test: dict,
    ):
        self.reconstructor = reconstructor
        self.X_test = X_test
        self.y_test = y_test

        # Store configuration from reconstructor
        self.NUM_LEPTONS = reconstructor.NUM_LEPTONS
        self.max_jets = reconstructor.max_jets
        self.met_features = reconstructor.met_features
        self.n_jets = reconstructor.n_jets
        self.n_leptons = reconstructor.n_leptons
        self.n_met = reconstructor.n_met
        self.padding_value = reconstructor.padding_value
        self.feature_indices = reconstructor.config.feature_indices

        # Initialize helper classes
        self.feature_importance_calc = FeatureImportanceCalculator(
            reconstructor, X_test, y_test
        )

    def plot_training_history(self):
        """Plot training and validation loss/accuracy over epochs."""
        if self.reconstructor.history is None:
            raise ValueError(
                "No training history found. Please train the model before "
                "plotting history."
            )

        history = self.reconstructor.history

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot loss
        self._plot_metric(
            axes[0],
            history,
            metric_name="loss",
            title="Model Loss",
            ylabel="Loss",
        )

        # Plot accuracy
        self._plot_metric(
            axes[1],
            history,
            metric_name="assignment_accuracy",
            title="Model Accuracy",
            ylabel="Accuracy",
        )

        fig.tight_layout()
        return fig, axes

    @staticmethod
    def _plot_metric(ax, history, metric_name: str, title: str, ylabel: str):
        """Helper method to plot a single training metric."""
        ax.plot(history.history[metric_name], label=f"Training {ylabel}")
        ax.plot(history.history[f"val_{metric_name}"], label=f"Validation {ylabel}")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()

    def compute_permutation_importance(self, num_repeats: int = 5) -> dict:
        """
        Compute feature importance using permutation importance.

        Args:
            num_repeats: Number of times to permute each feature

        Returns:
            Dictionary mapping feature names to importance scores
        """
        return self.feature_importance_calc.compute_permutation_importance(num_repeats)

    def plot_feature_importance(self, num_repeats: int = 10):
        """Plot feature importance scores."""
        importances = self.compute_permutation_importance(num_repeats=num_repeats)

        # Sort features by importance
        sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        features = [item[0] for item in sorted_items]
        scores = [item[1] for item in sorted_items]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(features, scores, color="skyblue")
        ax.set_xlabel("Importance Score")
        ax.set_title("Feature Importance based on Permutation Importance")
        ax.invert_yaxis()
        fig.tight_layout()

        return fig, ax
