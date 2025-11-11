"""Evaluator for comparing event reconstruction methods."""

import numpy as np
from typing import Union, Optional, List, Tuple

from . import EventReconstructorBase, GroundTruthReconstructor, MLReconstructorBase
from .evaluator_base import (
    PlotConfig,
    BootstrapCalculator,
    BinningUtility,
    FeatureExtractor,
    AccuracyCalculator,
    NeutrinoDeviationCalculator,
)
from .plotting_utils import (
    AccuracyPlotter,
    ConfusionMatrixPlotter,
    ComplementarityPlotter,
    ResolutionPlotter,
    NeutrinoDeviationPlotter,
)
from .physics_calculations import TopReconstructor, ResolutionCalculator


class PredictionManager:
    """Manages predictions from multiple reconstructors."""

    def __init__(
        self,
        reconstructors: List[EventReconstructorBase],
        X_test: dict,
    ):
        self.reconstructors = reconstructors
        self.X_test = X_test
        self.predictions = []
        self._compute_all_predictions()

    def _compute_all_predictions(self):
        """Compute predictions for all reconstructors."""
        for reconstructor in self.reconstructors:
            if isinstance(reconstructor, MLReconstructorBase):
                assignment_pred, neutrino_regression = reconstructor.complete_forward_pass(
                    self.X_test
                )
                self.predictions.append(
                    {
                        "assignment": assignment_pred,
                        "regression": neutrino_regression,
                    }
                )
            else:
                assignment_pred = reconstructor.predict_indices(self.X_test)
                if hasattr(reconstructor, "reconstruct_neutrinos"):
                    neutrino_pred = reconstructor.reconstruct_neutrinos(self.X_test)
                else:
                    print("WARNING: Reconstructor does not support neutrino regression.")
                    neutrino_pred = None
                self.predictions.append(
                    {
                        "assignment": assignment_pred,
                        "regression": neutrino_pred,
                    }
                )

    def get_assignment_predictions(self, index: int) -> np.ndarray:
        """Get assignment predictions for a specific reconstructor."""
        return self.predictions[index]["assignment"]

    def get_neutrino_predictions(self, index: int) -> np.ndarray:
        """Get neutrino predictions for a specific reconstructor."""
        return self.predictions[index]["regression"]


class ComplementarityAnalyzer:
    """Analyzes complementarity between reconstructors."""

    def __init__(
        self,
        prediction_manager: PredictionManager,
        y_test: dict,
    ):
        self.prediction_manager = prediction_manager
        self.y_test = y_test

    def compute_complementarity_matrix(self) -> np.ndarray:
        """
        Compute complementarity matrix between reconstructors.

        Returns:
            Complementarity matrix (n_reconstructors, n_reconstructors)
        """

        # Get per-event accuracy for each reconstructor
        per_event_accuracies = self._get_all_per_event_accuracies()

        n_reconstructors = len(per_event_accuracies)
        complementarity_matrix = np.zeros((n_reconstructors, n_reconstructors))

        # Compute complementarity
        for i in range(n_reconstructors):
            for j in range(n_reconstructors):
                if i != j:
                    both_incorrect = np.sum(
                        (1 - per_event_accuracies[i]) * (1 - per_event_accuracies[j])
                    )
                    total_events = len(per_event_accuracies[i])
                    complementarity_matrix[i, j] = both_incorrect / total_events

        return complementarity_matrix

    def get_per_event_complementarity(self) -> np.ndarray:
        """
        Get per-event success for at least one reconstructor.

        Returns:
            Array of per-event complementarity (n_events,)
        """
        per_event_accuracies = self._get_all_per_event_accuracies()
        per_event_success = np.stack(per_event_accuracies, axis=1).astype(bool)
        return np.any(per_event_success, axis=1).astype(float)

    def _get_all_per_event_accuracies(self) -> List[np.ndarray]:
        """Get per-event accuracies for all reconstructors."""
        accuracies = []
        for i in range(len(self.prediction_manager.reconstructors)):
            if isinstance(
                self.prediction_manager.reconstructors[i],
                GroundTruthReconstructor,
            ):
                continue
            predictions = self.prediction_manager.get_assignment_predictions(i)
            accuracy_data = AccuracyCalculator.compute_accuracy(
                self.y_test["assignment_labels"],
                predictions,
                per_event=True,
            )
            accuracies.append(accuracy_data)
        return accuracies


class ReconstructionEvaluator:
    """Evaluator for comparing event reconstruction methods."""

    def __init__(
        self,
        reconstructors: Union[List[EventReconstructorBase], EventReconstructorBase],
        X_test: dict,
        y_test: dict,
    ):
        # Handle single reconstructor
        if isinstance(reconstructors, EventReconstructorBase):
            reconstructors = [reconstructors]

        self.reconstructors = reconstructors
        self.X_test = X_test
        self.y_test = y_test

        # Validate configurations
        self._validate_configs()
        self.config = reconstructors[0].config

        # Setup neutrino momenta

        # Initialize managers
        self.prediction_manager = PredictionManager(reconstructors, X_test)
        self.complementarity_analyzer = ComplementarityAnalyzer(
            self.prediction_manager, y_test
        )

    def _validate_configs(self):
        """Validate that all reconstructors have the same configuration."""
        configs = [r.config for r in self.reconstructors]
        base_config = configs[0]

        for config in configs[1:]:
            if config != base_config:
                raise ValueError(
                    "All reconstructors must have the same DataConfig for "
                    "consistent evaluation."
                )

    # ==================== Accuracy Methods ====================

    def evaluate_accuracy(
        self,
        reconstructor_index: int,
        per_event: bool = False,
    ) -> Union[float, np.ndarray]:
        """Evaluate accuracy for a specific reconstructor."""
        predictions = self.prediction_manager.get_assignment_predictions(
            reconstructor_index
        )
        return AccuracyCalculator.compute_accuracy(
            self.y_test["assignment_labels"],
            predictions,
            per_event=per_event,
        )

    def _bootstrap_accuracy(
        self,
        reconstructor_index: int,
        config: PlotConfig,
    ) -> Tuple[float, float, float]:
        """Compute accuracy with bootstrap confidence intervals."""
        accuracy_data = self.evaluate_accuracy(reconstructor_index, per_event=True)
        return BootstrapCalculator.compute_bootstrap_ci(
            accuracy_data,
            n_bootstrap=config.n_bootstrap,
            confidence=config.confidence,
        )

    def plot_all_accuracies(
        self,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        figsize: Tuple[int, int] = (10, 6),
    ):
        """Plot accuracies for all reconstructors with error bars."""
        config = PlotConfig(
            figsize=figsize,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
        )

        print("\nComputing bootstrap confidence intervals...")
        accuracies = []

        names = []
        for i, reconstructor in enumerate(self.reconstructors):
            if isinstance(reconstructor, GroundTruthReconstructor):
                print(f"{reconstructor.get_name()}: Ground Truth (skipping)")
                continue
            mean_acc, lower, upper = self._bootstrap_accuracy(i, config)
            accuracies.append((mean_acc, lower, upper))
            print(
                f"{reconstructor.get_name()}: {mean_acc:.4f} "
                f"[{lower:.4f}, {upper:.4f}]"
            )
            names.append(reconstructor.get_name())

        return AccuracyPlotter.plot_overall_accuracies(names, accuracies, config)

    # ==================== Neutrino Deviation Methods ====================

    def evaluate_neutrino_deviation(
        self,
        reconstructor_index: int,
        per_event: bool = False,
        deviation_type: str = "relative",
    ) -> Union[float, np.ndarray]:
        """
        Evaluate neutrino reconstruction deviation for a specific reconstructor.

        Args:
            reconstructor_index: Index of the reconstructor
            per_event: If True, return per-event deviation; else overall mean
            deviation_type: Type of deviation ('relative' or 'absolute')

        Returns:
            Deviation value(s)
        """
        if self.y_test.get("regression_targets") is None:
            raise ValueError(
                "No regression targets found in y_test. "
                "Cannot evaluate neutrino deviation."
            )

        predictions = self.prediction_manager.get_neutrino_predictions(
            reconstructor_index
        )
        true_neutrinos = self.y_test["regression_targets"]

        if deviation_type == "relative":
            return NeutrinoDeviationCalculator.compute_relative_deviation(
                predictions,
                true_neutrinos,
                per_event=per_event,
            )
        elif deviation_type == "absolute":
            return NeutrinoDeviationCalculator.compute_absolute_deviation(
                predictions,
                true_neutrinos,
                per_event=per_event,
            )
        else:
            raise ValueError(
                f"Unknown deviation_type: {deviation_type}. "
                "Must be 'relative' or 'absolute'."
            )

    def _bootstrap_neutrino_deviation(
        self,
        reconstructor_index: int,
        config: PlotConfig,
        deviation_type: str = "relative",
    ) -> Tuple[float, float, float]:
        """Compute neutrino deviation with bootstrap confidence intervals."""
        deviation_data = self.evaluate_neutrino_deviation(
            reconstructor_index, per_event=True, deviation_type=deviation_type
        )
        return BootstrapCalculator.compute_bootstrap_ci(
            deviation_data,
            n_bootstrap=config.n_bootstrap,
            confidence=config.confidence,
        )

    def plot_all_neutrino_deviations(
        self,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        figsize: Tuple[int, int] = (10, 6),
        deviation_type: str = "relative",
    ):
        """
        Plot neutrino deviations for all reconstructors with error bars.

        Args:
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level for intervals
            figsize: Figure size
            deviation_type: Type of deviation ('relative' or 'absolute')

        Returns:
            Tuple of (figure, axis)
        """
        if self.y_test.get("regression_targets") is None:
            raise ValueError(
                "No regression targets found in y_test. "
                "Cannot evaluate neutrino deviation."
            )

        config = PlotConfig(
            figsize=figsize,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
        )

        print("\nComputing bootstrap confidence intervals for neutrino deviation...")
        deviations = []
        names = []

        for i, reconstructor in enumerate(self.reconstructors):
            if isinstance(reconstructor, GroundTruthReconstructor):
                print(f"{reconstructor.get_name()}: Ground Truth (skipping)")
                #continue

            # Check if reconstructor supports neutrino reconstruction
            neutrino_pred = self.prediction_manager.get_neutrino_predictions(i)
            if neutrino_pred is None:
                print(
                    f"{reconstructor.get_name()}: No neutrino reconstruction (skipping)"
                )
                continue

            mean_dev, lower, upper = self._bootstrap_neutrino_deviation(
                i, config, deviation_type
            )
            deviations.append((mean_dev, lower, upper))
            print(
                f"{reconstructor.get_name()}: {mean_dev:.4f} "
                f"[{lower:.4f}, {upper:.4f}]"
            )
            names.append(reconstructor.get_name())

        if not deviations:
            raise ValueError(
                "No reconstructors with neutrino reconstruction found. "
                "Cannot plot neutrino deviations."
            )

        return NeutrinoDeviationPlotter.plot_overall_deviations(
            names, deviations, config
        )

    # ==================== Binned Accuracy Methods ====================

    def plot_binned_accuracy(
        self,
        feature_data_type: str,
        feature_name: str,
        fancy_feature_label: Optional[str] = None,
        bins: int = 20,
        xlims: Optional[Tuple[float, float]] = None,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        show_errorbar: bool = True,
        show_combinatoric: bool = True,
    ):
        """Plot binned accuracy vs. a feature with bootstrap error bars."""
        config = PlotConfig(
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            show_errorbar=show_errorbar,
        )

        # Extract feature data
        feature_data = FeatureExtractor.extract_feature(
            self.X_test,
            self.config.feature_indices,
            feature_data_type,
            feature_name,
        )

        # Create bins
        bin_edges = BinningUtility.create_bins(feature_data, bins, xlims)
        binning_mask = BinningUtility.create_binning_mask(feature_data, bin_edges)
        bin_centers = BinningUtility.compute_bin_centers(bin_edges)

        # Get event weights
        event_weights = FeatureExtractor.get_event_weights(self.X_test)

        # Compute combinatoric baseline if requested
        combinatoric_accuracy = None
        if show_combinatoric:
            combinatoric_per_event = AccuracyCalculator.compute_combinatoric_baseline(
                self.X_test, self.config.padding_value
            )
            combinatoric_accuracy = BinningUtility.compute_weighted_binned_statistic(
                binning_mask, combinatoric_per_event, event_weights
            )

        # Compute binned accuracies for each reconstructor
        print(f"\nComputing binned accuracy for {feature_name}...")
        binned_accuracies = []
        names = []
        for i, reconstructor in enumerate(self.reconstructors):
            if isinstance(reconstructor, GroundTruthReconstructor):
                continue
            accuracy_data = self.evaluate_accuracy(i, per_event=True)

            if show_errorbar:
                mean_acc, lower, upper = BootstrapCalculator.compute_binned_bootstrap(
                    binning_mask,
                    event_weights,
                    accuracy_data,
                    config.n_bootstrap,
                    config.confidence,
                )
                binned_accuracies.append((mean_acc, lower, upper))
            else:
                binned_acc = BinningUtility.compute_weighted_binned_statistic(
                    binning_mask, accuracy_data, event_weights
                )
                binned_accuracies.append((binned_acc, binned_acc, binned_acc))
            names.append(reconstructor.get_name())
        # Compute bin counts
        bin_counts = np.sum(
            event_weights.reshape(1, -1) * binning_mask, axis=1
        ) / np.sum(event_weights)

        # Plot
        feature_label = fancy_feature_label or feature_name

        return AccuracyPlotter.plot_binned_accuracy(
            bin_centers,
            binned_accuracies,
            names,
            bin_counts,
            bin_edges,
            feature_label,
            config,
            show_combinatoric,
            combinatoric_accuracy,
        )

    def plot_feature_assignment_success(
        self,
        feature_data_type: str,
        feature_name: str,
        assigner_index: int,
        fancy_feature_label: Optional[str] = None,
        bins: int = 20,
        xlims: Optional[Tuple[float, float]] = None,
        figsize: Tuple[int, int] = (8, 6),
    ):
        """Plot feature distribution for correctly and incorrectly assigned events."""
        # Extract feature data
        feature_data = FeatureExtractor.extract_feature(
            self.X_test,
            self.config.feature_indices,
            feature_data_type,
            feature_name,
        )

        # Get event weights
        event_weights = FeatureExtractor.get_event_weights(self.X_test)

        # Get assignment success for the first reconstructor (as an example)
        assignment_success = self.evaluate_accuracy(assigner_index, per_event=True)

        # Create bins
        bin_edges = BinningUtility.create_bins(feature_data, bins, xlims)
        bin_centers = BinningUtility.compute_bin_centers(bin_edges)

        # Compute histograms
        correct_mask = assignment_success.astype(bool)
        incorrect_mask = ~correct_mask
        correct_weights = event_weights * correct_mask
        incorrect_weights = event_weights * incorrect_mask
        correct_hist = np.histogram(
            feature_data, bins=bin_edges, weights=correct_weights, density=True
        )[0]
        incorrect_hist = np.histogram(
            feature_data, bins=bin_edges, weights=incorrect_weights, density=True
        )[0]

        # Plot
        feature_label = fancy_feature_label or feature_name
        fig, ax = AccuracyPlotter.plot_feature_assignment_success(
            bin_centers,
            correct_hist,
            incorrect_hist,
            feature_label,
            figsize,
        )
        ax.set_title(
            f"{self.reconstructors[assigner_index].get_name()} Assignment Success vs {feature_label}"
        )
        return fig, ax

    def plot_top_mass_deviation_assignment_success(
        self,
        assigner_index: int,
        true_top_mass_labels: List[str] = ["truth_top_mass", "truth_tbar_mass"],
        fancy_feature_label: Optional[str] = None,
        bins: int = 20,
        xlims: Optional[Tuple[float, float]] = None,
        figsize: Tuple[int, int] = (8, 6),
    ):
        """Plot top mass deviation for correctly and incorrectly assigned events."""
        # Validate and extract true top masses
        true_top1_mass, true_top2_mass = self._extract_true_top_masses(
            true_top_mass_labels
        )

        # Compute predicted top masses
        top1_p4, top2_p4 = self._compute_top_lorentz_vectors(assigner_index)
        pred_top1_mass, pred_top2_mass = TopReconstructor.compute_top_masses(
            top1_p4, top2_p4
        )

        # Compute mass deviation (extended for both tops)
        mass_deviation = np.concatenate(
            [
                ResolutionCalculator.compute_relative_deviation(
                    pred_top1_mass, true_top1_mass
                ),
                ResolutionCalculator.compute_relative_deviation(
                    pred_top2_mass, true_top2_mass
                ),
            ]
        )

        # Get event weights and assignment success
        event_weights = FeatureExtractor.get_event_weights(self.X_test)
        assignment_success = self.evaluate_accuracy(assigner_index, per_event=True)
        assignment_success_extended = np.concatenate(
            [assignment_success, assignment_success]
        )
        event_weights = np.concatenate([event_weights, event_weights])

        # Create bins
        bin_edges = BinningUtility.create_bins(mass_deviation, bins, xlims)
        bin_centers = BinningUtility.compute_bin_centers(bin_edges)

        # Compute histograms
        correct_mask = assignment_success_extended.astype(bool)
        incorrect_mask = ~correct_mask
        correct_weights = event_weights * correct_mask
        incorrect_weights = event_weights * incorrect_mask
        correct_hist = np.histogram(
            mass_deviation, bins=bin_edges, weights=correct_weights, density=True
        )[0]
        incorrect_hist = np.histogram(
            mass_deviation, bins=bin_edges, weights=incorrect_weights, density=True
        )[0]

        # Plot
        feature_label = fancy_feature_label or "Relative Top Mass Deviation"
        fig, ax = AccuracyPlotter.plot_feature_assignment_success(
            bin_centers,
            correct_hist,
            incorrect_hist,
            feature_label,
            figsize,
        )
        ax.set_title(
            f"{self.reconstructors[assigner_index].get_name()} Assignment Success vs {feature_label}"
        )
        return fig, ax

    def plot_binned_neutrino_deviation(
        self,
        feature_data_type: str,
        feature_name: str,
        fancy_feature_label: Optional[str] = None,
        bins: int = 20,
        xlims: Optional[Tuple[float, float]] = None,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        show_errorbar: bool = True,
        deviation_type: str = "relative",
    ):
        """
        Plot binned neutrino deviation vs. a feature with bootstrap error bars.

        Args:
            feature_data_type: Type of feature data ('jet', 'lepton', 'met', etc.)
            feature_name: Name of the feature
            fancy_feature_label: Optional fancy label for the feature
            bins: Number of bins or bin edges
            xlims: Optional x-axis limits
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level for intervals
            show_errorbar: Whether to show error bars
            deviation_type: Type of deviation ('relative' or 'absolute')

        Returns:
            Tuple of (figure, axis)
        """
        if self.y_test.get("regression_targets") is None:
            raise ValueError(
                "No regression targets found in y_test. "
                "Cannot evaluate neutrino deviation."
            )

        config = PlotConfig(
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            show_errorbar=show_errorbar,
        )

        # Extract feature data
        feature_data = FeatureExtractor.extract_feature(
            self.X_test,
            self.config.feature_indices,
            feature_data_type,
            feature_name,
        )

        # Create bins
        bin_edges = BinningUtility.create_bins(feature_data, bins, xlims)
        binning_mask = BinningUtility.create_binning_mask(feature_data, bin_edges)
        bin_centers = BinningUtility.compute_bin_centers(bin_edges)

        # Get event weights
        event_weights = FeatureExtractor.get_event_weights(self.X_test)

        # Compute binned deviations for each reconstructor
        print(f"\nComputing binned neutrino deviation for {feature_name}...")
        binned_deviations = []
        names = []
        for i, reconstructor in enumerate(self.reconstructors):
            if isinstance(reconstructor, GroundTruthReconstructor):
                continue
            names.append(reconstructor.get_name())
            # Check if reconstructor supports neutrino reconstruction
            neutrino_pred = self.prediction_manager.get_neutrino_predictions(i)
            if neutrino_pred is None:
                print(
                    f"{reconstructor.get_name()}: No neutrino reconstruction (skipping)"
                )
                continue

            deviation_data = self.evaluate_neutrino_deviation(
                i, per_event=True, deviation_type=deviation_type
            )

            if show_errorbar:
                mean_dev, lower, upper = BootstrapCalculator.compute_binned_bootstrap(
                    binning_mask,
                    event_weights,
                    deviation_data,
                    config.n_bootstrap,
                    config.confidence,
                )
                binned_deviations.append((mean_dev, lower, upper))
            else:
                binned_dev = BinningUtility.compute_weighted_binned_statistic(
                    binning_mask, deviation_data, event_weights
                )
                binned_deviations.append((binned_dev, binned_dev, binned_dev))

        if not binned_deviations:
            raise ValueError(
                "No reconstructors with neutrino reconstruction found. "
                "Cannot plot binned neutrino deviations."
            )

        # Compute bin counts
        bin_counts = np.sum(
            event_weights.reshape(1, -1) * binning_mask, axis=1
        ) / np.sum(event_weights)

        # Plot
        feature_label = fancy_feature_label or feature_name

        return NeutrinoDeviationPlotter.plot_binned_deviation(
            bin_centers,
            binned_deviations,
            names,
            bin_counts,
            bin_edges,
            feature_label,
            config,
        )

    def plot_neutrino_component_deviations(
        self,
        bins: int = 50,
        xlims: Optional[Tuple[float, float]] = None,
        figsize: Tuple[int, int] = (15, 10),
        component_labels: List[str] = ["$p_x$", "$p_y$", "$p_z$"],
    ):
        """
        Plot histograms of relative deviation for each neutrino momentum component.

        Args:
            bins: Number of bins for histograms
            xlims: Optional x-axis limits (min, max)
            figsize: Figure size
            component_labels: Labels for momentum components

        Returns:
            Tuple of (figure, axes)
        """
        if self.y_test.get("regression_targets") is None:
            raise ValueError(
                "No regression targets found in y_test. "
                "Cannot evaluate neutrino component deviations."
            )

        # Get event weights
        event_weights = FeatureExtractor.get_event_weights(self.X_test)
        true_neutrinos = self.y_test["regression_targets"]

        # Collect predictions from all reconstructors
        predicted_neutrinos = []
        names = []

        for i, reconstructor in enumerate(self.reconstructors):
            if isinstance(reconstructor, GroundTruthReconstructor):
                continue

            # Check if reconstructor supports neutrino reconstruction
            neutrino_pred = self.prediction_manager.get_neutrino_predictions(i)
            if neutrino_pred is None:
                print(
                    f"{reconstructor.get_name()}: No neutrino reconstruction (skipping)"
                )
                continue

            predicted_neutrinos.append(neutrino_pred)
            names.append(reconstructor.get_name())

        if not predicted_neutrinos:
            raise ValueError(
                "No reconstructors with neutrino reconstruction found. "
                "Cannot plot component deviations."
            )

        return NeutrinoDeviationPlotter.plot_component_deviation_histograms(
            predicted_neutrinos,
            true_neutrinos,
            names,
            event_weights,
            bins,
            xlims,
            figsize,
            component_labels,
        )

    def plot_overall_neutrino_deviation_distribution(
        self,
        bins: int = 50,
        xlims: Optional[Tuple[float, float]] = None,
        figsize: Tuple[int, int] = (12, 6),
    ):
        """
        Plot histogram of overall relative deviation distribution for all reconstructors.

        Args:
            bins: Number of bins for histogram
            xlims: Optional x-axis limits (min, max)
            figsize: Figure size

        Returns:
            Tuple of (figure, axis)
        """
        if self.y_test.get("regression_targets") is None:
            raise ValueError(
                "No regression targets found in y_test. "
                "Cannot evaluate neutrino deviation distribution."
            )

        # Get event weights
        event_weights = FeatureExtractor.get_event_weights(self.X_test)
        true_neutrinos = self.y_test["regression_targets"]

        # Collect predictions from all reconstructors
        predicted_neutrinos = []
        names = []

        for i, reconstructor in enumerate(self.reconstructors):
            if isinstance(reconstructor, GroundTruthReconstructor):
                continue

            # Check if reconstructor supports neutrino reconstruction
            neutrino_pred = self.prediction_manager.get_neutrino_predictions(i)
            if neutrino_pred is None:
                print(
                    f"{reconstructor.get_name()}: No neutrino reconstruction (skipping)"
                )
                continue

            predicted_neutrinos.append(neutrino_pred)
            names.append(reconstructor.get_name())

        if not predicted_neutrinos:
            raise ValueError(
                "No reconstructors with neutrino reconstruction found. "
                "Cannot plot deviation distribution."
            )

        return NeutrinoDeviationPlotter.plot_overall_deviation_distribution(
            predicted_neutrinos,
            true_neutrinos,
            names,
            event_weights,
            bins,
            xlims,
            figsize,
        )

    # ==================== Confusion Matrix Methods ====================

    def plot_confusion_matrices(
        self,
        normalize: bool = True,
        figsize_per_plot: Tuple[int, int] = (5, 5),
    ):
        """Plot confusion matrices for all reconstructors."""
        predictions_list = [
            self.prediction_manager.get_assignment_predictions(i)
            for i in range(len(self.reconstructors))
            if not isinstance(
                self.reconstructors[i],
                GroundTruthReconstructor,
            )
        ]
        names = [r.get_name() for r in self.reconstructors if not isinstance(r, GroundTruthReconstructor)]

        return ConfusionMatrixPlotter.plot_confusion_matrices(
            self.y_test["assignment_labels"],
            predictions_list,
            names,
            normalize,
            figsize_per_plot,
        )

    # ==================== Complementarity Methods ====================

    def compute_complementarity_matrix(self) -> np.ndarray:
        """Compute complementarity matrix between reconstructors."""
        return self.complementarity_analyzer.compute_complementarity_matrix()

    def plot_complementarity_matrix(
        self,
        figsize: Tuple[int, int] = (8, 6),
    ):
        """Plot complementarity matrix between reconstructors."""
        matrix = self.compute_complementarity_matrix()
        names = [
            r.get_name()
            for r in self.reconstructors
            if not isinstance(r, GroundTruthReconstructor)
        ]
        return ComplementarityPlotter.plot_complementarity_matrix(
            matrix, names, figsize
        )

    def plot_binned_complementarity(
        self,
        feature_data_type: str,
        feature_name: str,
        fancy_feature_label: Optional[str] = None,
        bins: int = 20,
        xlims: Optional[Tuple[float, float]] = None,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        show_errorbar: bool = True,
    ):
        """Plot binned complementarity vs. a feature."""
        config = PlotConfig(
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            show_errorbar=show_errorbar,
        )

        # Extract feature data
        feature_data = FeatureExtractor.extract_feature(
            self.X_test,
            self.config.feature_indices,
            feature_data_type,
            feature_name,
        )

        # Create bins
        bin_edges = BinningUtility.create_bins(feature_data, bins, xlims)
        binning_mask = BinningUtility.create_binning_mask(feature_data, bin_edges)
        bin_centers = BinningUtility.compute_bin_centers(bin_edges)

        # Get event weights and complementarity
        event_weights = FeatureExtractor.get_event_weights(self.X_test)
        per_event_comp = self.complementarity_analyzer.get_per_event_complementarity()

        # Compute binned complementarity
        print(f"\nComputing binned complementarity for {feature_name}...")
        if show_errorbar:
            binned_comp = BootstrapCalculator.compute_binned_bootstrap(
                binning_mask,
                event_weights,
                per_event_comp,
                config.n_bootstrap,
                config.confidence,
            )
        else:
            binned_comp_mean = BinningUtility.compute_weighted_binned_statistic(
                binning_mask, per_event_comp, event_weights
            )
            binned_comp = (binned_comp_mean, binned_comp_mean, binned_comp_mean)

        # Compute bin counts
        bin_counts = np.sum(
            event_weights.reshape(1, -1) * binning_mask, axis=1
        ) / np.sum(event_weights)

        # Plot
        feature_label = fancy_feature_label or feature_name
        return ComplementarityPlotter.plot_binned_complementarity(
            bin_centers, binned_comp, bin_counts, bin_edges, feature_label, config
        )

    # ==================== Top Mass Resolution Methods ====================

    def _compute_top_lorentz_vectors(
        self,
        reconstructor_index: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute top reconstructions for a specific reconstructor."""
        assignment_pred = self.prediction_manager.get_assignment_predictions(
            reconstructor_index
        )
        neutrino_pred = self.prediction_manager.get_neutrino_predictions(
            reconstructor_index
        )

        lepton_features = self.X_test["lepton"][:, :, :4]
        jet_features = self.X_test["jet"][:, :, :4]

        return TopReconstructor.compute_top_lorentz_vectors(
            assignment_pred, neutrino_pred, lepton_features, jet_features
        )

    def plot_binned_top_mass_resolution(
        self,
        feature_data_type: str,
        feature_name: str,
        true_top_mass_labels: List[str] = ["truth_top_mass", "truth_tbar_mass"],
        fancy_feature_label: Optional[str] = None,
        bins: int = 20,
        xlims: Optional[Tuple[float, float]] = None,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        show_errorbar: bool = True,
    ):
        """Plot binned top mass resolution vs. a feature."""
        config = PlotConfig(
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            show_errorbar=show_errorbar,
        )

        # Validate and extract true top masses
        true_top1_mass, true_top2_mass = self._extract_true_top_masses(
            true_top_mass_labels
        )

        # Extract feature data and create bins
        feature_data = FeatureExtractor.extract_feature(
            self.X_test,
            self.config.feature_indices,
            feature_data_type,
            feature_name,
        )

        bin_edges = BinningUtility.create_bins(feature_data, bins, xlims)
        binning_mask = BinningUtility.create_binning_mask(feature_data, bin_edges)
        bin_centers = BinningUtility.compute_bin_centers(bin_edges)
        event_weights = FeatureExtractor.get_event_weights(self.X_test)

        # Compute resolutions for each reconstructor
        print(f"\nComputing binned top mass resolution for {feature_name}...")
        binned_resolutions = []

        for i, reconstructor in enumerate(self.reconstructors):
            # Compute top masses
            top1_p4, top2_p4 = self._compute_top_lorentz_vectors(i)
            top1_mass, top2_mass = TopReconstructor.compute_top_masses(top1_p4, top2_p4)

            # Compute deviations (extended for both tops)
            mass_deviation = np.concatenate(
                [
                    ResolutionCalculator.compute_relative_deviation(
                        top1_mass, true_top1_mass
                    ),
                    ResolutionCalculator.compute_relative_deviation(
                        top2_mass, true_top2_mass
                    ),
                ]
            )

            # Extend binning mask and weights
            binning_mask_extended = np.concatenate([binning_mask, binning_mask], axis=1)
            event_weights_extended = np.concatenate([event_weights, event_weights])

            if show_errorbar:
                mean_res, lower, upper = BootstrapCalculator.compute_binned_bootstrap(
                    binning_mask_extended,
                    event_weights_extended,
                    mass_deviation,
                    config.n_bootstrap,
                    config.confidence,
                )
                binned_resolutions.append((mean_res, lower, upper))
            else:
                resolution = BinningUtility.compute_weighted_binned_statistic(
                    binning_mask_extended,
                    mass_deviation,
                    event_weights_extended,
                    statistic="std",
                )
                binned_resolutions.append((resolution, resolution, resolution))

        # Compute bin counts
        bin_counts = np.sum(
            event_weights.reshape(1, -1) * binning_mask, axis=1
        ) / np.sum(event_weights)

        # Plot
        feature_label = fancy_feature_label or feature_name
        names = [r.get_name() for r in self.reconstructors]

        return ResolutionPlotter.plot_binned_resolution(
            bin_centers,
            binned_resolutions,
            names,
            bin_counts,
            bin_edges,
            feature_label,
            r"Relative $m(t)$ Deviation",
            config,
        )

    def plot_binned_ttbar_mass_resolution(
        self,
        feature_data_type: str,
        feature_name: str,
        true_ttbar_mass_label: List[str] = ["truth_ttbar_mass"],
        fancy_feature_label: Optional[str] = None,
        bins: int = 20,
        xlims: Optional[Tuple[float, float]] = None,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        show_errorbar: bool = True,
    ):
        """Plot binned ttbar mass resolution vs. a feature."""
        config = PlotConfig(
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            show_errorbar=show_errorbar,
        )

        # Validate and extract true ttbar mass
        true_ttbar_mass = self._extract_true_ttbar_mass(true_ttbar_mass_label)

        # Extract feature data and create bins
        feature_data = FeatureExtractor.extract_feature(
            self.X_test,
            self.config.feature_indices,
            feature_data_type,
            feature_name,
        )

        bin_edges = BinningUtility.create_bins(feature_data, bins, xlims)
        binning_mask = BinningUtility.create_binning_mask(feature_data, bin_edges)
        bin_centers = BinningUtility.compute_bin_centers(bin_edges)
        event_weights = FeatureExtractor.get_event_weights(self.X_test)

        # Compute resolutions for each reconstructor
        print(f"\nComputing binned ttbar mass resolution for {feature_name}...")
        binned_resolutions = []

        for i, reconstructor in enumerate(self.reconstructors):
            # Compute ttbar mass
            top1_p4, top2_p4 = self._compute_top_lorentz_vectors(i)
            ttbar_mass = TopReconstructor.compute_ttbar_mass(top1_p4, top2_p4)

            # Compute deviation
            mass_deviation = ResolutionCalculator.compute_relative_deviation(
                ttbar_mass, true_ttbar_mass
            )

            if show_errorbar:
                mean_res, lower, upper = BootstrapCalculator.compute_binned_bootstrap(
                    binning_mask,
                    event_weights,
                    mass_deviation,
                    config.n_bootstrap,
                    config.confidence,
                )
                binned_resolutions.append((mean_res, lower, upper))
            else:
                resolution = BinningUtility.compute_weighted_binned_statistic(
                    binning_mask, mass_deviation, event_weights, statistic="std"
                )
                binned_resolutions.append((resolution, resolution, resolution))

        # Compute bin counts
        bin_counts = np.sum(
            event_weights.reshape(1, -1) * binning_mask, axis=1
        ) / np.sum(event_weights)

        # Plot
        feature_label = fancy_feature_label or feature_name
        names = [r.get_name() for r in self.reconstructors]

        return ResolutionPlotter.plot_binned_resolution(
            bin_centers,
            binned_resolutions,
            names,
            bin_counts,
            bin_edges,
            feature_label,
            r"Relative $m(t\overline{t})$ Deviation",
            config,
        )

    def _extract_true_top_masses(
        self,
        labels: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract true top masses from test data."""
        for label in labels:
            if label not in self.config.feature_indices["non_training"]:
                raise ValueError(
                    f"True top mass label '{label}' not found in test data."
                )

        true_top1_mass = self.X_test["non_training"][
            :, self.config.feature_indices["non_training"][labels[0]]
        ]
        true_top2_mass = self.X_test["non_training"][
            :, self.config.feature_indices["non_training"][labels[1]]
        ]

        return true_top1_mass, true_top2_mass

    def _extract_true_ttbar_mass(self, labels: List[str]) -> np.ndarray:
        """Extract true ttbar mass from test data."""
        label = labels[0]
        if label not in self.config.feature_indices["non_training"]:
            raise ValueError(f"True ttbar mass label '{label}' not found in test data.")

        return self.X_test["non_training"][
            :, self.config.feature_indices["non_training"][label]
        ]
