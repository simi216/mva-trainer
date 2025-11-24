"""Evaluator for comparing event reconstruction methods."""

import numpy as np
from typing import Union, Optional, List, Tuple

from core.reconstruction import EventReconstructorBase, GroundTruthReconstructor, MLReconstructorBase
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
    SelectionAccuracyPlotter
)
from .physics_calculations import (
    TopReconstructor,
    ResolutionCalculator,
    lorentz_vector_from_PtEtaPhiE_array
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
        accuracy_data = self.evaluate_selection_accuracy(reconstructor_index, per_event=True)
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
                print(f"{reconstructor.get_name()}: Ground Truth (skipping)")
                continue
            mean_acc, lower, upper = self._bootstrap_selection_accuracy(i, config)
            selection_accuracies.append((mean_acc, lower, upper))
            print(
                f"{reconstructor.get_name()}: {mean_acc:.4f} "
                f"[{lower:.4f}, {upper:.4f}]"
            )
            names.append(reconstructor.get_name())

        return SelectionAccuracyPlotter.plot_selection_accuracies(names, selection_accuracies, config)



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
            combinatoric_per_event = SelectionAccuracyCalculator.compute_combinatoric_baseline(
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
            names.append(reconstructor.get_name())
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
        lepton_features = self.X_test["lepton"] # Assuming shape (n_events, n_leptons, n_features)

        # Get correctly assigned jet features
        jet_features = self.X_test["jet"][:,:, :4]  # Assuming first 4 features are kinematic (px, py, pz, E)
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

    def plot_binned_reconstructed_variable(
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
        combine_tops: bool = False,
        use_signed_deviation: bool = False,
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

        truth = self.compute_true_variable(
            truth_extractor, combine_tops=combine_tops
        )

        for i, reconstructor in enumerate(self.reconstructors):
            # Compute reconstructed and truth values
            reconstructed = self.compute_reconstructed_variable(
                i, variable_func, combine_tops=combine_tops
            )

            # Compute deviation
            if use_signed_deviation:
                deviation = ResolutionCalculator.compute_signed_relative_deviation(
                    reconstructed, truth
                )
            else:
                deviation = ResolutionCalculator.compute_relative_deviation(
                    reconstructed, truth
                )

            # Extend binning mask and weights if combining tops
            if combine_tops:
                binning_mask_ext = np.concatenate([binning_mask, binning_mask], axis=1)
                event_weights_ext = np.concatenate([event_weights, event_weights])
            else:
                binning_mask_ext = binning_mask
                event_weights_ext = event_weights

            if show_errorbar:
                mean_metric, lower, upper = BootstrapCalculator.compute_binned_bootstrap(
                    binning_mask_ext,
                    event_weights_ext,
                    deviation,
                    config.n_bootstrap,
                    config.confidence,
                    statistic=statistic,
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
        names = [r.get_name() for r in self.reconstructors]

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

    def plot_binned_top_mass_resolution(
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
        """Plot binned top mass resolution vs. a feature."""

        return self.plot_binned_reconstructed_variable(
            feature_data_type=feature_data_type,
            feature_name=feature_name,
            variable_func=self._compute_top_masses,
            truth_extractor=self._extract_true_top_masses,
            ylabel=r"Relative $m(t)$ Resolution",
            fancy_feature_label=fancy_feature_label,
            bins=bins,
            xlims=xlims,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            show_errorbar=show_errorbar,
            statistic="std",
            combine_tops=True,
            use_signed_deviation=False,
        )

    def plot_binned_ttbar_mass_resolution(
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
        """Plot binned ttbar mass resolution vs. a feature."""
        # Define variable computation function

        return self.plot_binned_reconstructed_variable(
            feature_data_type=feature_data_type,
            feature_name=feature_name,
            variable_func=self._compute_ttbar_mass,
            truth_extractor=self._extract_true_ttbar_mass,
            ylabel=r"Relative $m(t\overline{t})$ Resolution",
            fancy_feature_label=fancy_feature_label,
            bins=bins,
            xlims=xlims,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            show_errorbar=show_errorbar,
            statistic="std",
            combine_tops=False,
            use_signed_deviation=False,
        )

    def plot_binned_top_mass_deviation(
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
        """Plot binned top mass deviation (mean) vs. a feature."""

        return self.plot_binned_reconstructed_variable(
            feature_data_type=feature_data_type,
            feature_name=feature_name,
            variable_func=self._compute_top_masses,
            truth_extractor=self._extract_true_top_masses,
            ylabel=r"Mean Relative $m(t)$ Deviation",
            fancy_feature_label=fancy_feature_label,
            bins=bins,
            xlims=xlims,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            show_errorbar=show_errorbar,
            statistic="mean",
            combine_tops=True,
            use_signed_deviation=True,
        )

    def plot_binned_ttbar_mass_deviation(
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
        """Plot binned ttbar mass deviation (mean) vs. a feature."""

        return self.plot_binned_reconstructed_variable(
            feature_data_type=feature_data_type,
            feature_name=feature_name,
            variable_func=self._compute_ttbar_mass,
            truth_extractor=self._extract_true_ttbar_mass,
            ylabel=r"Mean Relative $m(t\overline{t})$ Deviation",
            fancy_feature_label=fancy_feature_label,
            bins=bins,
            xlims=xlims,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            show_errorbar=show_errorbar,
            statistic="mean",
            combine_tops=False,
            use_signed_deviation=True,
        )

    @staticmethod
    def _extract_true_top_masses(
        X_test,
    ) -> Tuple[np.ndarray, np.ndarray]:
        top_4_vector =lorentz_vector_from_PtEtaPhiE_array( X_test["top_truth"][:, 0, :4])
        tbar_4_vector = lorentz_vector_from_PtEtaPhiE_array(X_test["top_truth"][:, 1, :4])

        true_top_mass = TopReconstructor.compute_top_masses(
            top_4_vector, tbar_4_vector
        )
        return true_top_mass

    @staticmethod
    def _extract_true_ttbar_mass(X_test) -> np.ndarray:
        """Extract true ttbar mass from test data."""
        top_4_vector =lorentz_vector_from_PtEtaPhiE_array(X_test["top_truth"][:, 0, :4])
        tbar_4_vector = lorentz_vector_from_PtEtaPhiE_array(X_test["top_truth"][:, 1, :4])
        true_ttbar_mass = TopReconstructor.compute_ttbar_mass(
            top_4_vector, tbar_4_vector
        )
        return true_ttbar_mass

    @staticmethod
    def _compute_top_masses(
        leptons: np.ndarray,
        jets: np.ndarray,
        neutrinos: np.ndarray
    ):
        """Compute top and antitop masses from kinematic features."""
        top_p4, tbar_p4 = TopReconstructor.compute_top_lorentz_vectors(
            leptons,
            jets,
            neutrinos
        )
        return TopReconstructor.compute_top_masses(top_p4, tbar_p4)
    
    @staticmethod
    def _compute_ttbar_mass(
        leptons: np.ndarray,
        jets: np.ndarray,
        neutrinos: np.ndarray
    ):
        """Compute ttbar mass from kinematic features."""
        top_p4, tbar_p4 = TopReconstructor.compute_top_lorentz_vectors(
            leptons,
            jets,
            neutrinos
        )
        return TopReconstructor.compute_ttbar_mass(top_p4, tbar_p4)