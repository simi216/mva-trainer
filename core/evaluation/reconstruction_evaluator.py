"""Evaluator for comparing event reconstruction methods."""

import numpy as np
from typing import Union, Optional, List, Tuple
import matplotlib.pyplot as plt
import os
import timeit
from core.reconstruction import (
    EventReconstructorBase,
    GroundTruthReconstructor,
    KerasFFRecoBase,
)
from .evaluator_base import (
    PlotConfig,
    BootstrapCalculator,
    BinningUtility,
    FeatureExtractor,
    AccuracyCalculator,
    SelectionAccuracyCalculator,
    NeutrinoDeviationCalculator,
)
from .plotting_utils import (
    AccuracyPlotter,
    ConfusionMatrixPlotter,
    ComplementarityPlotter,
    ResolutionPlotter,
    NeutrinoDeviationPlotter,
    SelectionAccuracyPlotter,
    DistributionPlotter,
)
from .physics_calculations import (
    TopReconstructor,
    ResolutionCalculator,
    lorentz_vector_from_PtEtaPhiE_array,
    c_hel,
    c_han,
)


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
            if isinstance(reconstructor, KerasFFRecoBase):
                assignment_pred, neutrino_regression = (
                    reconstructor.complete_forward_pass(self.X_test)
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
                    print(
                        "WARNING: Reconstructor does not support neutrino regression."
                    )
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

class ComputationTimeEvaluator:
    """Evaluator for measuring computation time of reconstructors."""

    def __init__(
        self,
        reconstructors: List[EventReconstructorBase],
        X_test: dict,
    ):
        self.reconstructors = reconstructors
        self.X_test = X_test

    def evaluate_computation_times(self) -> List[float]:
        """Evaluate computation times for all reconstructors."""
        times = []
        for reconstructor in self.reconstructors:
            start_time = timeit.default_timer()
            reconstructor.predict_indices(self.X_test)
            end_time = timeit.default_timer()
            times.append(end_time - start_time)
        return times

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

    def evaluate_selection_accuracy(
        self,
        reconstructor_index: int,
        per_event: bool = False,
    ) -> Union[float, np.ndarray]:
        """Evaluate selection accuracy for a specific reconstructor."""
        predictions = self.prediction_manager.get_assignment_predictions(
            reconstructor_index
        )
        return SelectionAccuracyCalculator.compute_selection_accuracy(
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

    def _bootstrap_selection_accuracy(
        self,
        reconstructor_index: int,
        config: PlotConfig,
    ) -> Tuple[float, float, float]:
        """Compute selection accuracy with bootstrap confidence intervals."""
        accuracy_data = self.evaluate_selection_accuracy(
            reconstructor_index, per_event=True
        )
        return BootstrapCalculator.compute_bootstrap_ci(
            accuracy_data,
            n_bootstrap=config.n_bootstrap,
            confidence=config.confidence,
        )

    def plot_all_accuracies(
        self,
        n_bootstrap: int = 100,
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
                print(f"{reconstructor.get_assignment_name()}: Ground Truth (skipping)")
                continue
            mean_acc, lower, upper = self._bootstrap_accuracy(i, config)
            accuracies.append((mean_acc, lower, upper))
            print(
                f"{reconstructor.get_assignment_name()}: {mean_acc:.4f} "
                f"[{lower:.4f}, {upper:.4f}]"
            )
            names.append(reconstructor.get_assignment_name())

        return AccuracyPlotter.plot_overall_accuracies(names, accuracies, config)

    def plot_all_selection_accuracies(
        self,
        n_bootstrap: int = 100,
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
        selection_accuracies = []

        names = []
        for i, reconstructor in enumerate(self.reconstructors):
            if isinstance(reconstructor, GroundTruthReconstructor):
                print(f"{reconstructor.get_assignment_name()}: Ground Truth (skipping)")
                continue
            mean_acc, lower, upper = self._bootstrap_selection_accuracy(i, config)
            selection_accuracies.append((mean_acc, lower, upper))
            print(
                f"{reconstructor.get_assignment_name()}: {mean_acc:.4f} "
                f"[{lower:.4f}, {upper:.4f}]"
            )
            names.append(reconstructor.get_assignment_name())

        return SelectionAccuracyPlotter.plot_selection_accuracies(
            names, selection_accuracies, config
        )

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
        if self.y_test.get("neutrino_truth") is None:
            raise ValueError(
                "No regression targets found in y_test. "
                "Cannot evaluate neutrino deviation."
            )

        predictions = self.prediction_manager.get_neutrino_predictions(
            reconstructor_index
        )
        true_neutrinos = self.y_test["neutrino_truth"]

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
        n_bootstrap: int = 100,
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
        if self.y_test.get("neutrino_truth") is None:
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
                print(f"{reconstructor.get_full_reco_name()}: Ground Truth (skipping)")
                # continue

            # Check if reconstructor supports neutrino reconstruction
            neutrino_pred = self.prediction_manager.get_neutrino_predictions(i)
            if neutrino_pred is None:
                print(
                    f"{reconstructor.get_full_reco_name()}: No neutrino reconstruction (skipping)"
                )
                continue

            mean_dev, lower, upper = self._bootstrap_neutrino_deviation(
                i, config, deviation_type
            )
            deviations.append((mean_dev, lower, upper))
            print(
                f"{reconstructor.get_full_reco_name()}: {mean_dev:.4f} "
                f"[{lower:.4f}, {upper:.4f}]"
            )
            names.append(reconstructor.get_full_reco_name())

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
        n_bootstrap: int = 100,
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
            names.append(reconstructor.get_assignment_name())
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

    def plot_binned_selection_accuracy(
        self,
        feature_data_type: str,
        feature_name: str,
        fancy_feature_label: Optional[str] = None,
        bins: int = 20,
        xlims: Optional[Tuple[float, float]] = None,
        n_bootstrap: int = 100,
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
            combinatoric_per_event = (
                SelectionAccuracyCalculator.compute_combinatoric_baseline(
                    self.X_test, self.config.padding_value
                )
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
            accuracy_data = self.evaluate_selection_accuracy(i, per_event=True)

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
            names.append(reconstructor.get_assignment_name())
        # Compute bin counts
        bin_counts = np.sum(
            event_weights.reshape(1, -1) * binning_mask, axis=1
        ) / np.sum(event_weights)

        # Plot
        feature_label = fancy_feature_label or feature_name

        return SelectionAccuracyPlotter.plot_binned_selection_accuracy(
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
        names = [
            r.get_assignment_name()
            for r in self.reconstructors
            if not isinstance(r, GroundTruthReconstructor)
        ]

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
            r.get_assignment_name()
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
        n_bootstrap: int = 100,
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

    def compute_reconstructed_variable(
        self,
        reconstructor_index: int,
        variable_func: callable,
        combine_tops: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute any reconstructed variable from event kinematics.

        Args:
            reconstructor_index: Index of the reconstructor
            variable_func: Function that takes (top1_p4, top2_p4, lepton_features, jet_features, neutrino_pred)
                          and returns the reconstructed variable(s)
            truth_extractor: Optional function to extract truth values from X_test
            combine_tops: If True, concatenate results for both tops (for per-top quantities)

        Returns:
            Reconstructed variable array or tuple of (reconstructed, truth) if truth_extractor provided
        """
        # Get predictions
        assignment_pred = self.prediction_manager.get_assignment_predictions(
            reconstructor_index
        )
        neutrino_pred = self.prediction_manager.get_neutrino_predictions(
            reconstructor_index
        )

        # Get lepton features
        lepton_features = self.X_test[
            "lepton"
        ]  # Assuming shape (n_events, n_leptons, n_features)

        # Get correctly assigned jet features
        jet_features = self.X_test["jet"][
            :, :, :4
        ]  # Assuming first 4 features are kinematic (Pt, Eta, Phi, E)
        selected_jet_indices = assignment_pred.argmax(axis=-2)
        reco_jets = np.take_along_axis(
            jet_features,
            selected_jet_indices[:, :, np.newaxis],
            axis=1,
        )

        # Compute reconstructed variable
        reconstructed = variable_func(lepton_features, reco_jets, neutrino_pred)

        # Handle per-top quantities
        if combine_tops and isinstance(reconstructed, tuple):
            reconstructed = np.concatenate(reconstructed)

        return reconstructed

    def compute_true_variable(
        self,
        truth_extractor: callable,
        combine_tops: bool = False,
    ) -> np.ndarray:
        """
        Compute any truth variable from X_test.

        Args:
            truth_extractor: Function to extract truth values from X_test
            combine_tops: If True, concatenate results for both tops (for per-top quantities)
        Returns:
            Truth variable array
        """
        truth = truth_extractor(self.X_test)

        # Handle per-top quantities
        if combine_tops and isinstance(truth, tuple):
            truth = np.concatenate(truth)

        return truth

    def plot_binned_reco_resolution(
        self,
        feature_data_type: str,
        feature_name: str,
        variable_func: callable,
        truth_extractor: callable,
        ylabel: str,
        fancy_feature_label: Optional[str] = None,
        bins: int = 20,
        xlims: Optional[Tuple[float, float]] = None,
        n_bootstrap: int = 100,
        confidence: float = 0.95,
        show_errorbar: bool = True,
        statistic: str = "std",
        use_signed_deviation: bool = False,
        use_relative_deviation: bool = True,
        combine_tops: bool = False,
    ):
        """
        Plot binned resolution or deviation of any reconstructed variable vs. a feature.

        Args:
            feature_data_type: Type of feature data ('jet', 'lepton', 'met', etc.)
            feature_name: Name of the feature to bin by
            variable_func: Function that takes (top1_p4, top2_p4, lepton_features, jet_features, neutrino_pred)
                          and returns reconstructed variable(s)
            truth_extractor: Function to extract truth values from (X_test, feature_indices)
            ylabel: Y-axis label for the plot
            fancy_feature_label: Optional fancy label for the feature
            bins: Number of bins
            xlims: Optional x-axis limits
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level for intervals
            show_errorbar: Whether to show error bars
            statistic: Statistic to compute ('std' for resolution, 'mean' for deviation)
            combine_tops: If True, combine results from both tops (for per-top quantities)
            use_signed_deviation: If True, use signed deviation instead of absolute

        Returns:
            Tuple of (figure, axis)
        """
        config = PlotConfig(
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            show_errorbar=show_errorbar,
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

        # Compute metric for each reconstructor
        print(f"\nComputing binned {ylabel} for {feature_name}...")
        binned_metrics = []

        for i, reconstructor in enumerate(self.reconstructors):
            # Compute reconstructed and truth values
            reconstructed = self.compute_reconstructed_variable(
                i, variable_func, combine_tops=combine_tops
            )
            truth = self.compute_true_variable(
                truth_extractor, combine_tops=combine_tops
            )

            # Compute deviation
            deviation = ResolutionCalculator.compute_deviation(
                reconstructed,
                truth,
                use_signed_deviation=use_signed_deviation,
                use_relative_deviation=use_relative_deviation,
            )

            # Extend binning mask and weights if combining tops
            if combine_tops:
                binning_mask_ext = np.concatenate([binning_mask, binning_mask], axis=1)
                event_weights_ext = np.concatenate([event_weights, event_weights])
            else:
                binning_mask_ext = binning_mask
                event_weights_ext = event_weights

            if show_errorbar:
                mean_metric, lower, upper = (
                    BootstrapCalculator.compute_binned_bootstrap(
                        binning_mask_ext,
                        event_weights_ext,
                        deviation,
                        config.n_bootstrap,
                        config.confidence,
                        statistic=statistic,
                    )
                )
                binned_metrics.append((mean_metric, lower, upper))
            else:
                metric = BinningUtility.compute_weighted_binned_statistic(
                    binning_mask_ext,
                    deviation,
                    event_weights_ext,
                    statistic=statistic,
                )
                binned_metrics.append((metric, metric, metric))

        # Compute bin counts
        bin_counts = np.sum(
            event_weights.reshape(1, -1) * binning_mask, axis=1
        ) / np.sum(event_weights)

        # Plot
        feature_label = fancy_feature_label or feature_name
        names = [r.get_full_reco_name() for r in self.reconstructors]

        return ResolutionPlotter.plot_binned_resolution(
            bin_centers,
            binned_metrics,
            names,
            bin_counts,
            bin_edges,
            feature_label,
            ylabel,
            config,
        )

    def plot_reco_vs_truth_distribution(
        self,
        ax,
        reconstructor_index: int,
        variable_func: callable,
        truth_extractor: callable,
        variable_label: str,
        bins: int = 50,
        xlims: Optional[Tuple[float, float]] = None,
        combine_tops: bool = False,
    ):
        """
        Plot distribution of reconstructed variable vs. truth.

        Args:
            reconstructor_index: Index of the reconstructor
            variable_func: Function that takes (top1_p4, top2_p4, lepton_features, jet_features, neutrino_pred)
                          and returns reconstructed variable(s)
            truth_extractor: Function to extract truth values from X_test
            variable_label: Label for the variable being plotted
            bins: Number of bins
            xlims: Optional x-axis limits
            figsize: Figure size

        Returns:
            Tuple of (figure, axis)
        """
        # Compute reconstructed and truth values
        reconstructed = self.compute_reconstructed_variable(
            reconstructor_index, variable_func
        )
        truth = self.compute_true_variable(truth_extractor)
        event_weights = FeatureExtractor.get_event_weights(self.X_test)

        if combine_tops:
            reconstructed = np.concatenate(reconstructed)
            truth = np.concatenate(truth)
            event_weights = np.concatenate([event_weights, event_weights])
        return DistributionPlotter.plot_feature_distributions(
            [reconstructed, truth],
            variable_label,
            event_weights=event_weights,
            bins=bins,
            xlims=xlims,
            labels=["reco", "truth"],
            ax=ax,
        )

    def plot_deviations_distributions(
        self,
        ax,
        reconstructor_index: int,
        variable_func: callable,
        truth_extractor: callable,
        variable_label: str,
        use_relative_deviation: bool = True,
        use_signed_deviation: bool = True,
        combine_tops: bool = False,
        **kwargs,
    ):
        """Plot distribution of deviations between reconstructed variable and truth."""
        # Compute reconstructed and truth values
        reconstructed = self.compute_reconstructed_variable(
            reconstructor_index, variable_func
        )
        truth = self.compute_true_variable(truth_extractor)
        event_weights = FeatureExtractor.get_event_weights(self.X_test)

        # Compute deviation
        deviation = ResolutionCalculator.compute_deviation(
            reconstructed,
            truth,
            use_signed_deviation=use_signed_deviation,
            use_relative_deviation=use_relative_deviation,
        )
        if combine_tops:
            deviation = np.concatenate(deviation)
            event_weights = np.concatenate([event_weights, event_weights])

        return DistributionPlotter.plot_feature_distributions(
            [deviation],
            f"Deviation in {variable_label}",
            event_weights=event_weights,
            labels=[self.reconstructors[reconstructor_index].get_full_reco_name()],
            ax=ax,
            **kwargs,
        )

    def plot_deviations_distributions_all_reconstructors(
        self,
        variable_func: callable,
        truth_extractor: callable,
        variable_label: str,
        figsize: Optional[Tuple[int, int]] = (10, 10),
        **kwargs,
    ):
        """
        Plot distributions of deviations for all reconstructors.

        Args:
            variable_func: Function that computes the variable from (leptons, jets, neutrinos)
            truth_extractor: Function that extracts truth values from X_test
            variable_label: Label for the variable being plotted
            xlims: Optional x-axis limits
            bins: Number of bins
            figsize: Figure size
        Returns:
            Tuple of (figure, axes)
        """
        fig, axes = plt.subplots(
            figsize=figsize,
        )

        # Collect all deviations and labels
        all_deviations = []
        labels = []
        event_weights = FeatureExtractor.get_event_weights(self.X_test)

        # Extract common parameters from kwargs
        use_relative_deviation = kwargs.pop("use_relative_deviation", True)
        use_signed_deviation = kwargs.pop("use_signed_deviation", True)
        combine_tops = kwargs.pop("combine_tops", False)

        reco_index = 0
        for reconstructor in self.reconstructors:

            # Compute reconstructed and truth values
            reconstructed = self.compute_reconstructed_variable(
                reco_index, variable_func
            )
            truth = self.compute_true_variable(truth_extractor)

            # Compute deviation
            deviation = ResolutionCalculator.compute_deviation(
                reconstructed,
                truth,
                use_signed_deviation=use_signed_deviation,
                use_relative_deviation=use_relative_deviation,
            )
            if combine_tops:
                deviation = np.concatenate(deviation)

            all_deviations.append(deviation)
            labels.append(reconstructor.get_full_reco_name())
            reco_index += 1

        # Handle combined tops for event weights
        if combine_tops:
            event_weights_plot = np.concatenate([event_weights, event_weights])
        else:
            event_weights_plot = event_weights

        # Plot all deviations together
        DistributionPlotter.plot_feature_distributions(
            all_deviations,
            f"Deviation in {variable_label}",
            event_weights=event_weights_plot,
            labels=labels,
            ax=axes,
            **kwargs,
        )

        axes.set_title(f"{variable_label} Deviation for all Reconstructors")

        return fig, axes

    def plot_distributions_all_reconstructors(
        self,
        variable_func: callable,
        truth_extractor: callable,
        variable_label: str,
        xlims: Optional[Tuple[float, float]] = None,
        bins: int = 50,
        figsize: Optional[Tuple[int, int]] = (10, 10),
        save_individual_plots: bool = False,
        **kwargs,
    ):
        """
        Plot distributions for all reconstructors and truth.

        Args:
            variable_func: Function that computes the variable from (leptons, jets, neutrinos)
            truth_extractor: Function that extracts truth values from X_test
            variable_label: Label for the variable being plotted
            xlims: Optional x-axis limits
            bins: Number of bins
            figsize: Figure size

        Returns:
            Tuple of (figure, axes)
        """
        num_plots = len(self.reconstructors)  # Exclude ground truth

        num_cols = np.ceil(np.sqrt(num_plots)).astype(int)
        num_rows = np.ceil(num_plots / num_cols).astype(int)

        if not save_individual_plots:
            fig, axes = plt.subplots(
                num_rows,
                num_cols,
                figsize=figsize,
                constrained_layout=True,
            )
            axes = axes.flatten()
        else:
            fig, axes = [], []
            for i in range(num_plots):
                fig_i, ax_i = plt.subplots(
                    figsize=figsize,
                    constrained_layout=True,
                )
                fig.append(fig_i)
                axes.append(ax_i)

        reco_index = 0
        for reconstructor in self.reconstructors:
            #            if isinstance(reconstructor, GroundTruthReconstructor):
            #                continue
            ax = axes[reco_index]
            self.plot_reco_vs_truth_distribution(
                ax,
                reco_index,
                variable_func,
                truth_extractor,
                variable_label,
                bins=bins,
                xlims=xlims,
                **kwargs,
            )
            reco_index += 1
            ax.set_title(reconstructor.get_full_reco_name())

        if not save_individual_plots:
            for i in range(reco_index, len(axes)):
                fig.delaxes(axes[i])  # Remove unused subplots

        return fig, axes

    # ==================== Plot Specific Variable distributions ====================

    def _plot_variable_distribution(self, variable_key: str, **kwargs):
        """Generic method to plot variable distributions using configuration."""
        config = self._get_variable_config(variable_key)
        return self.plot_distributions_all_reconstructors(
            variable_func=config["compute_func"],
            truth_extractor=config["extract_func"],
            variable_label=config["label"],
            combine_tops=config.get("combine_tops", False),
            **kwargs,
        )

    def plot_c_hel_distributions(self, **kwargs):
        """Plot cos_hel distributions for all reconstructors and truth."""
        return self._plot_variable_distribution("c_hel", **kwargs)

    def plot_c_han_distributions(self, **kwargs):
        """Plot cos_han distributions for all reconstructors and truth."""
        return self._plot_variable_distribution("c_han", **kwargs)

    def plot_top_mass_distributions(self, **kwargs):
        """Plot top mass distributions for all reconstructors and truth."""

        fig, axes = self._plot_variable_distribution("top_mass", **kwargs)
        for ax in axes:
            ticks = ax.get_xticks()
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{tick/1000:.1f}" for tick in ticks])  # Convert to TeV
        return fig, axes

    def plot_ttbar_mass_distributions(self, **kwargs):
        """Plot ttbar mass distributions for all reconstructors and truth."""
        fig, axes = self._plot_variable_distribution("ttbar_mass", **kwargs)
        for ax in axes:
            ticks = ax.get_xticks()
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{tick/1000:.1f}" for tick in ticks])  # Convert to TeV
        return fig, axes

    # ==================== Deviation Distribution Methods ====================

    def _plot_variable_deviation(self, variable_key: str, **kwargs):
        """Generic method to plot variable deviations using configuration."""
        config = self._get_variable_config(variable_key)
        # Set defaults from config, allow kwargs to override
        defaults = {
            "use_relative_deviation": config.get("use_relative_deviation", False),
            "combine_tops": config.get("combine_tops", False),
        }
        defaults.update(kwargs)

        return self.plot_deviations_distributions_all_reconstructors(
            variable_func=config["compute_func"],
            truth_extractor=config["extract_func"],
            variable_label=f"{config['label']}",
            **defaults,
        )

    def plot_top_mass_deviation_distribution(self, **kwargs):
        """Plot top mass deviation distribution for all reconstructors."""
        kwargs.setdefault("use_relative_deviation", True)
        kwargs.setdefault("combine_tops", True)
        return self._plot_variable_deviation("top_mass", **kwargs)

    def plot_ttbar_mass_deviation_distribution(self, **kwargs):
        """Plot ttbar mass deviation distribution for all reconstructors."""
        return self._plot_variable_deviation("ttbar_mass", **kwargs)

    def plot_c_han_deviation_distribution(self, **kwargs):
        """Plot cos_han deviation distribution for all reconstructors."""
        return self._plot_variable_deviation("c_han", **kwargs)

    def plot_c_hel_deviation_distribution(self, **kwargs):
        """Plot cos_hel deviation distribution for all reconstructors."""
        return self._plot_variable_deviation("c_hel", **kwargs)

    # ==================== Binned Variable Resolution/Deviation Methods ====================

    def _plot_binned_variable(
        self,
        variable_key: str,
        metric_type: str,
        feature_data_type: str,
        feature_name: str,
        **kwargs,
    ):
        """Generic method to plot binned metrics using configuration.

        Args:
            variable_key: Key identifying the variable (e.g., 'top_mass', 'c_han')
            metric_type: Either 'resolution' or 'deviation'
            feature_data_type: Type of feature data for binning
            feature_name: Name of feature for binning
            **kwargs: Additional arguments passed to plot_binned_reco_resolution
        """
        config = self._get_variable_config(variable_key)
        resolution_config = config.get("resolution", {})

        # Determine parameters based on metric type
        if metric_type == "resolution":
            statistic = "std"
            use_signed_deviation = False
            ylabel_template = resolution_config.get(
                "ylabel_resolution", f"{config['label']} Resolution"
            )
        else:  # deviation
            statistic = "mean"
            use_signed_deviation = True
            ylabel_template = resolution_config.get(
                "ylabel_deviation", f"Mean {config['label']} Deviation"
            )

        # Get defaults from config
        defaults = {
            "statistic": statistic,
            "use_signed_deviation": use_signed_deviation,
            "use_relative_deviation": resolution_config.get(
                "use_relative_deviation", True
            ),
            "combine_tops": config.get("combine_tops", False),
        }
        defaults.update(kwargs)

        return self.plot_binned_reco_resolution(
            feature_data_type=feature_data_type,
            feature_name=feature_name,
            variable_func=config["compute_func"],
            truth_extractor=config["extract_func"],
            ylabel=ylabel_template,
            **defaults,
        )

    def plot_binned_top_mass_resolution(
        self, feature_data_type: str, feature_name: str, **kwargs
    ):
        """Plot binned top mass resolution vs. a feature."""
        return self._plot_binned_variable(
            "top_mass", "resolution", feature_data_type, feature_name, **kwargs
        )

    def plot_binned_ttbar_mass_resolution(
        self, feature_data_type: str, feature_name: str, **kwargs
    ):
        """Plot binned ttbar mass resolution vs. a feature."""
        return self._plot_binned_variable(
            "ttbar_mass", "resolution", feature_data_type, feature_name, **kwargs
        )

    def plot_binned_top_mass_deviation(
        self, feature_data_type: str, feature_name: str, **kwargs
    ):
        """Plot binned top mass deviation (mean) vs. a feature."""
        return self._plot_binned_variable(
            "top_mass", "deviation", feature_data_type, feature_name, **kwargs
        )

    def plot_binned_ttbar_mass_deviation(
        self, feature_data_type: str, feature_name: str, **kwargs
    ):
        """Plot binned ttbar mass deviation (mean) vs. a feature."""
        return self._plot_binned_variable(
            "ttbar_mass", "deviation", feature_data_type, feature_name, **kwargs
        )

    def plot_binned_c_han_resolution(
        self, feature_data_type: str, feature_name: str, **kwargs
    ):
        """Plot binned cos_han resolution vs. a feature."""
        return self._plot_binned_variable(
            "c_han", "resolution", feature_data_type, feature_name, **kwargs
        )

    def plot_binned_c_hel_resolution(
        self, feature_data_type: str, feature_name: str, **kwargs
    ):
        """Plot binned cos_hel resolution vs. a feature."""
        return self._plot_binned_variable(
            "c_hel", "resolution", feature_data_type, feature_name, **kwargs
        )

    def plot_binned_c_han_deviation(
        self, feature_data_type: str, feature_name: str, **kwargs
    ):
        """Plot binned cos_han deviation (mean) vs. a feature."""
        return self._plot_binned_variable(
            "c_han", "deviation", feature_data_type, feature_name, **kwargs
        )

    def plot_binned_c_hel_deviation(
        self, feature_data_type: str, feature_name: str, **kwargs
    ):
        """Plot binned cos_hel deviation (mean) vs. a feature."""
        return self._plot_binned_variable(
            "c_hel", "deviation", feature_data_type, feature_name, **kwargs
        )

    def plot_binned_neutrino_magnitude_resolution(
        self, feature_data_type: str, feature_name: str, **kwargs
    ):
        """Plot binned neutrino magnitude resolution vs. a feature."""
        return self._plot_binned_variable(
            "neutrino_mag", "resolution", feature_data_type, feature_name, **kwargs
        )
    
    def plot_binned_neutrino_magnitude_deviation(
        self, feature_data_type: str, feature_name: str, **kwargs
    ):
        """Plot binned neutrino magnitude deviation (mean) vs. a feature."""
        return self._plot_binned_variable(
            "neutrino_mag", "deviation", feature_data_type, feature_name, **kwargs
        )


    # ==================== Variable Configuration and Computation Methods ====================

    @staticmethod
    def _get_variable_config(variable_key: str) -> dict:
        """Get configuration for a specific physics variable.

        Returns a dict with:
            - compute_func: Function to compute variable from reconstructed kinematics
            - extract_func: Function to extract truth value from X_test
            - label: LaTeX label for plotting
            - combine_tops: Whether to combine both tops for this variable
            - use_relative_deviation: Whether to use relative deviation
            - resolution: Dict with ylabel templates for resolution/deviation plots
        """
        configs = {
            "top_mass": {
                "compute_func": lambda l, j, n: TopReconstructor.compute_top_masses(
                    *TopReconstructor.compute_top_lorentz_vectors(l, j, n)
                ),
                "extract_func": lambda X: TopReconstructor.compute_top_masses(
                    lorentz_vector_from_PtEtaPhiE_array(X["top_truth"][:, 0, :4]),
                    lorentz_vector_from_PtEtaPhiE_array(X["top_truth"][:, 1, :4]),
                ),
                "label": r"$m(t)$ [GeV]",
                "combine_tops": True,
                "use_relative_deviation": True,
                "resolution": {
                    "use_relative_deviation": True,
                    "ylabel_resolution": r"Relative $m(t)$ Resolution",
                    "ylabel_deviation": r"Mean Relative $m(t)$ Deviation",
                },
            },
            "ttbar_mass": {
                "compute_func": lambda l, j, n: TopReconstructor.compute_ttbar_mass(
                    *TopReconstructor.compute_top_lorentz_vectors(l, j, n)
                ),
                "extract_func": lambda X: TopReconstructor.compute_ttbar_mass(
                    lorentz_vector_from_PtEtaPhiE_array(X["top_truth"][:, 0, :4]),
                    lorentz_vector_from_PtEtaPhiE_array(X["top_truth"][:, 1, :4]),
                ),
                "label": r"$m(t\overline{t})$ [GeV]",
                "combine_tops": False,
                "use_relative_deviation": True,
                "resolution": {
                    "use_relative_deviation": True,
                    "ylabel_resolution": r"Relative $m(t\overline{t})$ Resolution",
                    "ylabel_deviation": r"Mean Relative $m(t\overline{t})$ Deviation",
                },
            },
            "c_han": {
                "compute_func": lambda l, j, n: c_han(
                    *TopReconstructor.compute_top_lorentz_vectors(l, j, n),
                    lorentz_vector_from_PtEtaPhiE_array(l[:, 0, :4]),
                    lorentz_vector_from_PtEtaPhiE_array(l[:, 1, :4]),
                ),
                "extract_func": lambda X: c_han(
                    lorentz_vector_from_PtEtaPhiE_array(X["top_truth"][:, 0, :4]),
                    lorentz_vector_from_PtEtaPhiE_array(X["top_truth"][:, 1, :4]),
                    lorentz_vector_from_PtEtaPhiE_array(X["lepton_truth"][:, 0, :4]),
                    lorentz_vector_from_PtEtaPhiE_array(X["lepton_truth"][:, 1, :4]),
                ),
                "label": r"$c_{\text{han}}$",
                "combine_tops": False,
                "use_relative_deviation": False,
                "resolution": {
                    "use_relative_deviation": False,
                    "ylabel_resolution": r"$c_{\text{han}}$ Resolution",
                    "ylabel_deviation": r"Mean $c_{\text{han}}$ Deviation",
                },
            },
            "c_hel": {
                "compute_func": lambda l, j, n: c_hel(
                    *TopReconstructor.compute_top_lorentz_vectors(l, j, n),
                    lorentz_vector_from_PtEtaPhiE_array(l[:, 0, :4]),
                    lorentz_vector_from_PtEtaPhiE_array(l[:, 1, :4]),
                ),
                "extract_func": lambda X: c_hel(
                    lorentz_vector_from_PtEtaPhiE_array(X["top_truth"][:, 0, :4]),
                    lorentz_vector_from_PtEtaPhiE_array(X["top_truth"][:, 1, :4]),
                    lorentz_vector_from_PtEtaPhiE_array(X["lepton_truth"][:, 0, :4]),
                    lorentz_vector_from_PtEtaPhiE_array(X["lepton_truth"][:, 1, :4]),
                ),
                "label": r"$c_{\text{hel}}$",
                "combine_tops": False,
                "use_relative_deviation": False,
                "resolution": {
                    "use_relative_deviation": False,
                    "ylabel_resolution": r"$c_{\text{hel}}$ Resolution",
                    "ylabel_deviation": r"Mean $c_{\text{hel}}$ Deviation",
                },
            },
            "neutrino_mag": {
                "compute_func": lambda l, j, n: (np.linalg.norm(n[:,0,:], axis=-1),np.linalg.norm(n[:,1,:], axis=-1)),
                "extract_func": lambda X: (np.linalg.norm(X["neutrino_truth"][:,0,:], axis=-1),np.linalg.norm(X["neutrino_truth"][:,1,:], axis=-1)),
                "label": r"$|\vec{p}(\nu)|$ [GeV]",
                "combine_tops": True,
                "use_relative_deviation": True,
                "resolution": {
                    "use_relative_deviation": True,
                    "ylabel_resolution": r"Relative $|\vec{p}(\nu)|$ Resolution",
                    "ylabel_deviation": r"Mean Relative $|\vec{p}(\nu)|$ Deviation",
                },
            },
        }

        if variable_key not in configs:
            raise ValueError(f"Unknown variable key: {variable_key}")
        return configs[variable_key]

    def save_accuracy_latex_table(
        self,
        n_bootstrap: int = 100,
        confidence: float = 0.95,
        caption: str = "Reconstruction Accuracies",
        label: str = "tab:accuracies",
        save_dir: Optional[str] = None,
    ) -> str:
        """
        Generate LaTeX table with accuracy and selection accuracy for all reconstructors.

        Args:
            n_bootstrap: Number of bootstrap samples for confidence intervals
            confidence: Confidence level for intervals
            caption: Table caption
            label: Table label for referencing

        Returns:
            LaTeX table string
        """
        config = PlotConfig(
            n_bootstrap=n_bootstrap,
            confidence=confidence,
        )

        # Collect results
        results = []
        for i, reconstructor in enumerate(self.reconstructors):
            if isinstance(reconstructor, GroundTruthReconstructor):
                continue

            name = reconstructor.get_assignment_name()

            # Compute accuracy with CI
            acc_mean, acc_lower, acc_upper = self._bootstrap_accuracy(i, config)

            # Compute selection accuracy with CI
            sel_acc_mean, sel_acc_lower, sel_acc_upper = (
                self._bootstrap_selection_accuracy(i, config)
            )

            results.append(
                {
                    "name": name,
                    "accuracy": (acc_mean, acc_lower, acc_upper),
                    "selection_accuracy": (sel_acc_mean, sel_acc_lower, sel_acc_upper),
                }
            )

        # Generate LaTeX table
        latex = []
        latex.append(r"    \begin{tabular}{lcc}")
        latex.append(r"        \toprule")
        latex.append(r"        Method & Assignment Accuracy & Selection Accuracy \\")
        latex.append(r"        \midrule")

        for res in results:
            name = res["name"]
            acc_mean, acc_lower, acc_upper = res["accuracy"]
            sel_mean, sel_lower, sel_upper = res["selection_accuracy"]

            acc_str = (
                f"${acc_mean:.4f}" +"_{-" +f"{acc_mean - acc_lower:.4f}"+ "}" +"^{+"+f"{acc_upper - acc_mean:.4f}" + "}$"
            )
            sel_str = (
                f"${sel_mean:.4f}" +"_{-" +f"{sel_mean - sel_lower:.4f}"+ "}" +"^{+"+f"{sel_upper - sel_mean:.4f}" + "}$"
            )

            latex.append(f"        {name} & {acc_str} & {sel_str} \\\\")

        latex.append(r"        \bottomrule")
        latex.append(r"    \end{tabular}")

        latex_str = "\n".join(latex)
        file_name = "reconstruction_accuracies_table.tex"
        if save_dir is not None:
            file_name = os.path.join(save_dir, file_name)
        with open(file_name, "w") as f:
            f.write(latex_str)
        print(f"LaTeX table saved to {file_name}")


    def plot_binned_accuracy_quotients(
        self,
        feature_data_type: str,
        feature_name: str,
        fancy_feature_label: Optional[str] = None,
        bins: int = 20,
        xlims: Optional[Tuple[float, float]] = None,
        n_bootstrap: int = 100,
        confidence: float = 0.95,
    ):
        """
        Plot binned quotient of assignment accuracy / selection accuracy vs. a feature.

        The quotient indicates how well the assignment performs relative to just
        selecting the correct jets (regardless of assignment to leptons).

        Args:
            feature_data_type: Type of feature data ('jet', 'lepton', 'met', etc.)
            feature_name: Name of the feature to bin by
            fancy_feature_label: Optional fancy label for the feature
            bins: Number of bins
            xlims: Optional x-axis limits
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level for intervals
            show_errorbar: Whether to show error bars

        Returns:
            Tuple of (figure, axis)
        """
        config = PlotConfig(
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            show_errorbar=True,
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

        # Compute binned quotients for each reconstructor
        print(f"\nComputing binned accuracy quotients for {feature_name}...")
        binned_quotients = []
        names = []

        for i, reconstructor in enumerate(self.reconstructors):
            if isinstance(reconstructor, GroundTruthReconstructor):
                continue

            # Get per-event accuracies
            assignment_acc = self.evaluate_accuracy(i, per_event=True)
            selection_acc = self.evaluate_selection_accuracy(i, per_event=True)

            mean_quotient, lower, upper = BootstrapCalculator.compute_binned_function_bootstrap(
                binning_mask,
                event_weights,
                (assignment_acc, selection_acc),
                lambda x, y: np.divide(x,y, out=np.zeros_like(x), where=y != 0),
                config.n_bootstrap,
                config.confidence,
                statistic="mean",
            )
            binned_quotients.append((mean_quotient, lower, upper))

            names.append(reconstructor.get_assignment_name())

        # Compute bin counts
        bin_counts = np.sum(
            event_weights.reshape(1, -1) * binning_mask, axis=1
        ) / np.sum(event_weights)

        # Plot
        feature_label = fancy_feature_label or feature_name
        fig, ax = plt.subplots(figsize=(10, 6))
        color_map = plt.get_cmap("tab10")
        fmt_map = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', 'h', '8']

        for i, (name, (mean, lower, upper)) in enumerate(zip(names, binned_quotients)):
            ax.errorbar(
                bin_centers,
                mean,
                yerr=[mean - lower, upper - mean],
                fmt=fmt_map[i % len(fmt_map)],
                color=color_map(i % 10),
                label=name,
                linestyle="None",
            )

        ax.plot(
            bin_centers, 0.5 * np.ones_like(bin_centers), linestyle="--", color="black", label="Random Assignment"
        )

        ax.set_xlabel(feature_label)
        ax.set_ylabel('Assignment Accuracy / Selection Accuracy')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title(f"Binned Accuracy Quotients vs {feature_label} ({config.confidence*100:.0f}% CI)")
        AccuracyPlotter._add_count_histogram(
            ax, bin_centers, bin_counts, bin_edges
        )
        plt.tight_layout()

        return fig, ax
    
    def plot_relative_neutrino_deviations(self, bins = 10, xlims = None):
        """
        Plot deviation distributions for magnitude and direction of neutrino momenta
                
        :param bins: Number of bins
        :param xlims: Optional x-axis limits
        """
        fig, ax = plt.subplots(figsize=(10,5*self.config.NUM_LEPTONS), ncols=2, nrows=self.config.NUM_LEPTONS)
        true_neutrino = self.y_test["neutrino_truth"]
        true_neutrino_mag = np.linalg.norm(true_neutrino[...,:3], axis=-1)
        true_neutrino_dir = true_neutrino[...,:3] / true_neutrino_mag[..., np.newaxis]
        event_weights = FeatureExtractor.get_event_weights(self.X_test)
        neutrino_deviations = []
        names = []
        nu_flows = False
        for i, reconstructor in enumerate(self.reconstructors):
            if isinstance(reconstructor, GroundTruthReconstructor):
                continue
            if reconstructor.use_nu_flows and not nu_flows:
                nu_flows = True
                names.append(r"$\nu^2$-Flows")
            elif reconstructor.use_nu_flows and nu_flows:
                continue
            else:
                names.append(reconstructor.get_full_reco_name())
            pred_neutrino = self.prediction_manager.get_neutrino_predictions(i)
            pred_neutrino_mag = np.linalg.norm(pred_neutrino[...,:3], axis=-1)
            pred_neutrino_dir = pred_neutrino[...,:3] / pred_neutrino_mag[..., np.newaxis]
            # Compute deviations
            mag_deviation = (pred_neutrino_mag - true_neutrino_mag )/ true_neutrino_mag
            dir_deviation = np.arccos(np.clip(np.sum(pred_neutrino_dir * true_neutrino_dir, axis=-1), -1.0, 1.0))
            neutrino_deviations.append(np.array([mag_deviation, dir_deviation]))

        for lepton_idx in range(self.config.NUM_LEPTONS):
            for comp_idx, component in enumerate([r"$\Delta |\vec{p}|$", r"$\Delta \phi$"]):
                ax_i = ax[lepton_idx, comp_idx]
                DistributionPlotter.plot_feature_distributions(
                    [
                        neutrino_deviations[i][comp_idx][..., lepton_idx]
                        for i in range(len(neutrino_deviations))
                    ],
                    f"{component}" + r"$(\nu_{" + f"{lepton_idx+1}" + r"})$",
                    event_weights=event_weights,
                    labels=names,
                    bins=bins,
                    xlims=xlims,
                    ax=ax_i,
                )

        # Collect handles and labels from the first axis
        handles, labels = ax[0, 0].get_legend_handles_labels()
        
        # Remove individual legends from all axes
        for lepton_idx in range(self.config.NUM_LEPTONS):
            for comp_idx in range(2):
                ax[lepton_idx, comp_idx].get_legend().remove()
        
        # Add single legend for the whole figure
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=len(names))
    
        plt.tight_layout()
        return fig, ax