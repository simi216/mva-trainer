"""Physics calculations for event reconstruction."""

import numpy as np
from typing import Tuple, Optional

from core.utils import (
    lorentz_vector_from_PtEtaPhiE_array,
    lorentz_vector_from_neutrino_momenta_array,
    compute_mass_from_lorentz_vector_array,
)


class TopReconstructor:
    """Handles reconstruction of top quark kinematics."""

    @staticmethod
    def compute_top_lorentz_vectors(
        assignment_predictions: np.ndarray,
        neutrino_predictions: np.ndarray,
        lepton_features: np.ndarray,
        jet_features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute top quark four-vectors from predictions.

        Args:
            assignment_predictions: Predicted jet assignments (n_events, n_leptons, n_jets)
            neutrino_predictions: Predicted neutrino momenta (n_events, 2, 3)
            lepton_features: Lepton features (n_events, n_leptons, n_features)
            jet_features: Jet features (n_events, n_jets, n_features)

        Returns:
            Tuple of (top1_p4, top2_p4) four-vectors
        """
        # Select jets based on predictions
        selected_jet_indices = assignment_predictions.argmax(axis=-2)
        reco_jets = np.take_along_axis(
            jet_features,
            selected_jet_indices[:, :, np.newaxis],
            axis=1,
        )

        # Reshape leptons and neutrinos
        reco_leptons = lepton_features.reshape(-1, 2, 4)
        reco_neutrinos = neutrino_predictions.reshape(-1, 2, 3)

        # Convert to four-vectors
        reco_jets_p4 = lorentz_vector_from_PtEtaPhiE_array(reco_jets)
        reco_leptons_p4 = lorentz_vector_from_PtEtaPhiE_array(reco_leptons)
        reco_neutrinos_p4 = lorentz_vector_from_neutrino_momenta_array(reco_neutrinos)

        # Compute top four-vectors
        top1_p4 = reco_jets_p4[:, 0] + reco_leptons_p4[:, 0] + reco_neutrinos_p4[:, 0]
        top2_p4 = reco_jets_p4[:, 1] + reco_leptons_p4[:, 1] + reco_neutrinos_p4[:, 1]

        return top1_p4, top2_p4

    @staticmethod
    def compute_top_masses(
        top1_p4: np.ndarray,
        top2_p4: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute top quark masses from four-vectors.

        Args:
            top1_p4: Top 1 four-vector
            top2_p4: Top 2 four-vector

        Returns:
            Tuple of (top1_mass, top2_mass)
        """
        top1_mass = compute_mass_from_lorentz_vector_array(top1_p4)
        top2_mass = compute_mass_from_lorentz_vector_array(top2_p4)
        return top1_mass, top2_mass

    @staticmethod
    def compute_ttbar_mass(
        top1_p4: np.ndarray,
        top2_p4: np.ndarray,
    ) -> np.ndarray:
        """
        Compute ttbar system mass.

        Args:
            top1_p4: Top 1 four-vector
            top2_p4: Top 2 four-vector

        Returns:
            ttbar mass array
        """
        ttbar_p4 = top1_p4 + top2_p4
        return compute_mass_from_lorentz_vector_array(ttbar_p4)


class ResolutionCalculator:
    """Calculate mass resolution metrics."""

    @staticmethod
    def compute_relative_deviation(
        predicted_mass: np.ndarray,
        true_mass: np.ndarray,
    ) -> np.ndarray:
        """Compute relative mass deviation."""
        return np.abs(predicted_mass - true_mass) / true_mass

    @staticmethod
    def compute_signed_relative_deviation(
        predicted_mass: np.ndarray,
        true_mass: np.ndarray,
    ) -> np.ndarray:
        """Compute signed relative mass deviation."""
        return (predicted_mass - true_mass) / true_mass

    @staticmethod
    def compute_resolution(
        deviations: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute resolution as standard deviation of deviations.

        Args:
            deviations: Array of relative deviations
            weights: Optional event weights

        Returns:
            Resolution value
        """
        if weights is None:
            weights = np.ones_like(deviations)

        mean_deviation = np.average(deviations, weights=weights)
        variance = np.average(
            (deviations - mean_deviation) ** 2,
            weights=weights,
        )
        return np.sqrt(variance)

    @staticmethod
    def compute_binned_resolution(
        deviations: np.ndarray,
        binning_mask: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Compute resolution in bins.

        Args:
            deviations: Array of relative deviations
            binning_mask: Boolean mask for binning (n_bins, n_events)
            weights: Event weights

        Returns:
            Array of resolutions per bin
        """
        n_bins = binning_mask.shape[0]
        resolutions = np.zeros(n_bins)

        for i in range(n_bins):
            bin_mask = binning_mask[i]
            if np.sum(bin_mask) > 0:
                bin_deviations = deviations[bin_mask]
                bin_weights = weights[bin_mask]
                resolutions[i] = ResolutionCalculator.compute_resolution(
                    bin_deviations, bin_weights
                )

        return resolutions
