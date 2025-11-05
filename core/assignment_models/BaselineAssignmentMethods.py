from core.reconstruction import EventReconstructorBase
from core.DataLoader import DataConfig
import numpy as np
from core.utils.four_vector_arithmetics import (
    lorentz_vector_from_pt_eta_phi_e,
    compute_mass_from_lorentz_vector,
)


class BaselineAssigner(EventReconstructorBase):
    def __init__(self, config: DataConfig, name="baseline_assigner", mode="min"):
        super().__init__(config, name)
        """Initializes the BaselineAssigner class.
        Args:
            config (DataConfig): Configuration object containing data parameters.
            mode (str): The mode of assignment, either "min" or "max".
        """
        if mode not in ["min", "max"]:
            raise ValueError("Mode must be either 'min' or 'max'.")
        if config.has_regression_targets:
            raise ValueError("BaselineAssigner does not support regression targets.")
        self.mode = mode
        self.max_leptons = config.max_leptons
        self.max_jets = config.max_jets

    def compute_comparison_feature(self, data_dict):
        """
        Computes a comparison feature between leptons and jets.
        This is a placeholder implementation and should be replaced with actual logic.

        Args:
            data_dict (dict): A dictionary containing input data for the model.
        Returns:
            np.ndarray: A 3D array of shape (num_events, max_jets,max_leptons) representing the comparison feature.
        """
        raise NotImplementedError(
            "This method should be implemented with actual logic."
        )

    def get_jets_mask(self, data_dict):
        padding_value = self.config.padding_value
        jet_mask = (data_dict["jet"] != padding_value).all(
            axis=-1
        )  # Shape: (num_events, max_jets)

        jet_mask = np.repeat(jet_mask[:, :, np.newaxis], self.max_leptons, axis=2)
        return jet_mask

    def get_viable_jets_mask(self, data_dict):
        """
        Computes a mask indicating viable jets for each lepton based on the padding value.

        Args:
            data_dict (dict): A dictionary containing input data for the model.
        Returns:
            np.ndarray: A 3D boolean array of shape (num_events, max_leptons, max_jets) indicating viable jets.
        """
        jet_mask = self.get_jets_mask(data_dict)
        jet_b_tag = None
        jet_pt = None
        for feature in self.feature_index_dict["jet"]:
            if "btag" in feature.lower() or "b_tag" in feature.lower():
                jet_b_tag = data_dict["jet"][
                    :, :, self.feature_index_dict["jet"][feature]
                ]
            if "pt" in feature.lower():
                jet_pt = data_dict["jet"][:, :, self.feature_index_dict["jet"][feature]]
        if jet_b_tag is not None:
            b_tag_mask = jet_b_tag >= 2 & jet_mask[:, :, 0]
            less_than_2_b_tag_jet_mask = np.sum(b_tag_mask) < 2
            iteration = 0
            while less_than_2_b_tag_jet_mask.any():
                jet_pt[b_tag_mask] = -1
                leading_jet_pt_indices = np.argmax(jet_pt, axis=1)
                leading_jet_pt_indices = leading_jet_pt_indices[:, np.newaxis]
                print(np.sum(update_mask, axis=1))
                update_mask = less_than_2_b_tag_jet_mask[:, np.newaxis] & (
                    np.arange(jet_mask.shape[1]) == leading_jet_pt_indices
                )
                b_tag_mask |= update_mask
                less_than_2_b_tag_jet_mask = np.sum(b_tag_mask, axis=1) < 2
                iteration += 1
                if iteration > self.max_jets:
                    raise RuntimeError(
                        "Exceeded maximum iterations while ensuring at least 2 b-tagged jets."
                    )
                    break
            jet_mask = jet_mask & b_tag_mask[:, :, np.newaxis]
        return jet_mask

    def predict_indices(self, data_dict):
        """
        Predicts the indices of jets assigned to each lepton based on a comparison feature.

        Args:
            data_dict (dict): A dictionary containing input data for the model.
        Returns:
            np.ndarray: A 3D array of shape (num_events, max_jets,max_leptons) representing the predicted indices.
        """
        comparison_feature = self.compute_comparison_feature(data_dict)
        viable_jets_mask = self.get_viable_jets_mask(data_dict)
        num_events = comparison_feature.shape[0]
        predicted_indices = np.zeros((num_events, self.max_jets, self.max_leptons))
        for i in range(num_events):
            for j in range(self.max_leptons):
                valid_indices = np.where(viable_jets_mask[i, :, j])[0]
                if valid_indices.size > 0:
                    if self.mode == "min":
                        best_jet_index = valid_indices[
                            np.argmin(comparison_feature[i, valid_indices, j])
                        ]
                    else:  # self.mode == "max"
                        best_jet_index = valid_indices[
                            np.argmax(comparison_feature[i, valid_indices, j])
                        ]
                    predicted_indices[i, best_jet_index, j] = 1
                    viable_jets_mask[i, best_jet_index, :] = (
                        False  # Exclude this jet for other leptons
                    )
        return predicted_indices


class DeltaRAssigner(BaselineAssigner):
    def __init__(
        self, config: DataConfig, name="delta_r_assigner", mode="min", b_tag_threshold=2
    ):
        super().__init__(config, name, mode)
        """Initializes the DeltaRAssigner class.
        Args:
            config (DataConfig): Configuration object containing data parameters.
            mode (str): The mode of assignment, either "min" or "max".
        """
        self.padding_value = config.padding_value
        self.lepton_features = config.lepton_features
        self.jet_features = config.jet_features
        self.feature_index_dict = config.feature_indices
        self.b_tag_threshold = b_tag_threshold

    def compute_comparison_feature(self, data_dict):
        """
        Computes the Delta R between leptons and jets.

        Args:
            data_dict (dict): A dictionary containing input data for the model.
        Returns:
            np.ndarray: A 3D array of shape (num_events, max_jets, max_leptons) representing the Delta R values.
        """
        leptons = data_dict["lepton"]
        jets = data_dict["jet"]
        lepton_eta = None
        lepton_phi = None
        for feature in self.lepton_features:
            if "eta" in feature.lower():
                lepton_eta = leptons[:, :, self.feature_index_dict["lepton"][feature]]
            if "phi" in feature.lower():
                lepton_phi = leptons[:, :, self.feature_index_dict["lepton"][feature]]
        jet_eta = None
        jet_phi = None
        for feature in self.jet_features:
            if "eta" in feature.lower():
                jet_eta = jets[:, :, self.feature_index_dict["jet"][feature]]
            if "phi" in feature.lower():
                jet_phi = jets[:, :, self.feature_index_dict["jet"][feature]]
        if (
            lepton_eta is None
            or lepton_phi is None
            or jet_eta is None
            or jet_phi is None
        ):
            raise ValueError(
                "Eta and Phi features must be present in both leptons and jets."
            )
        delta_eta = jet_eta[:, :, np.newaxis] - lepton_eta[:, np.newaxis, :]
        delta_phi = jet_phi[:, :, np.newaxis] - lepton_phi[:, np.newaxis, :]
        delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi  # Wrap-around
        delta_r = np.sqrt(delta_eta**2 + delta_phi**2)
        return delta_r


class LeptonJetMassAssigner(BaselineAssigner):
    def __init__(self, config: DataConfig, name="lepton_jet_mass_assigner", mode="min"):
        super().__init__(config, name, mode)
        """Initializes the LeptonJetMassAssigner class.
        Args:
            config (DataConfig): Configuration object containing data parameters.
            mode (str): The mode of assignment, either "min" or "max".
        """
        self.padding_value = config.padding_value
        self.lepton_features = config.lepton_features
        self.jet_features = config.jet_features
        self.feature_index_dict = config.feature_indices

    def compute_comparison_feature(self, data_dict):
        """
        Computes the Delta R between leptons and jets.

        Args:
            data_dict (dict): A dictionary containing input data for the model.
        Returns:
            np.ndarray: A 3D array of shape (num_events, max_jets,max_leptons) representing the Delta R values.
        """
        leptons = data_dict["lepton"]
        jets = data_dict["jet"]
        lepton_energy = leptons[:, :, self.feature_index_dict["lepton"]["lep_e"]][
            :, np.newaxis, :
        ]
        lepton_pt = leptons[:, :, self.feature_index_dict["lepton"]["lep_pt"]][
            :, np.newaxis, :
        ]
        lepton_eta = leptons[:, :, self.feature_index_dict["lepton"]["lep_eta"]][
            :, np.newaxis, :
        ]
        lepton_phi = leptons[:, :, self.feature_index_dict["lepton"]["lep_phi"]][
            :, np.newaxis, :
        ]
        jet_energy = jets[:, :, self.feature_index_dict["jet"]["ordered_jet_e"]][
            :, :, np.newaxis
        ]
        jet_pt = jets[:, :, self.feature_index_dict["jet"]["ordered_jet_pt"]][
            :, :, np.newaxis
        ]
        jet_eta = jets[:, :, self.feature_index_dict["jet"]["ordered_jet_eta"]][
            :, :, np.newaxis
        ]
        jet_phi = jets[:, :, self.feature_index_dict["jet"]["ordered_jet_phi"]][
            :, :, np.newaxis
        ]
        lep_px, lep_py, lep_pz, lep_e = lorentz_vector_from_pt_eta_phi_e(
            lepton_pt, lepton_eta, lepton_phi, lepton_energy
        )
        jet_px, jet_py, jet_pz, jet_e = lorentz_vector_from_pt_eta_phi_e(
            jet_pt, jet_eta, jet_phi, jet_energy
        )
        combined_px = lep_px + jet_px
        combined_py = lep_py + jet_py
        combined_pz = lep_pz + jet_pz
        combined_e = lep_e + jet_e
        invariant_mass = compute_mass_from_lorentz_vector(
            combined_px, combined_py, combined_pz, combined_e
        )
        return invariant_mass


class MassCombinatoricsAssigner(EventReconstructorBase):
    """Assigns jets to leptons based on mass combinatorics involving neutrino momenta."""
    def __init__(
        self,
        config: DataConfig,
        neutrino_momenta_branches: list,
        top_mass=173.15e3,
        name="mass_combinatorics_assigner",
        all_jets_considered=False,
    ):
        super().__init__(config, name)
        """Initializes the MassCombinatoricsAssigner class.
        Args:
            config (DataConfig): Configuration object containing data parameters.
        """
        self.max_leptons = config.max_leptons
        self.feature_index_dict = config.feature_indices
        self.max_jets = config.max_jets
        if len(neutrino_momenta_branches) != 6:
            raise ValueError(
                "neutrino_momenta_branches must contain exactly 6 elements."
            )
        if not all(
            branch in config.feature_indices["non_training"]
            for branch in neutrino_momenta_branches
        ):
            raise ValueError(
                "All neutrino_momenta_branches must be present in non_training_features."
            )
        self.neutrino_momenta_branches = neutrino_momenta_branches
        self.top_mass = top_mass
        self.all_jets_considered = all_jets_considered

    def get_jets_mask(self, data_dict):
        padding_value = self.config.padding_value
        jet_mask = (data_dict["jet"] != padding_value).all(
            axis=-1
        )  # Shape: (num_events, max_jets)

        jet_mask = np.repeat(jet_mask[:, :, np.newaxis], self.max_leptons, axis=2)
        return jet_mask

    def get_viable_jets_mask(self, data_dict):
        """
        Computes a mask indicating viable jets for each lepton based on the padding value.

        Args:
            data_dict (dict): A dictionary containing input data for the model.
        Returns:
            np.ndarray: A 3D boolean array of shape (num_events, max_leptons, max_jets) indicating viable jets.
        """
        jet_mask = self.get_jets_mask(data_dict)
        if self.all_jets_considered:
            return jet_mask
        jet_b_tag = None
        jet_pt = None
        for feature in self.feature_index_dict["jet"]:
            if "btag" in feature.lower() or "b_tag" in feature.lower():
                jet_b_tag = data_dict["jet"][
                    :, :, self.feature_index_dict["jet"][feature]
                ]
            if "pt" in feature.lower():
                jet_pt = data_dict["jet"][:, :, self.feature_index_dict["jet"][feature]]
        if jet_b_tag is not None:
            b_tag_mask = jet_b_tag >= 2 & jet_mask[:, :, 0]
            less_than_2_b_tag_jet_mask = np.sum(b_tag_mask) < 2
            iteration = 0
            while less_than_2_b_tag_jet_mask.any():
                jet_pt[b_tag_mask] = -1
                leading_jet_pt_indices = np.argmax(jet_pt, axis=1)
                leading_jet_pt_indices = leading_jet_pt_indices[:, np.newaxis]
                print(np.sum(update_mask, axis=1))
                update_mask = less_than_2_b_tag_jet_mask[:, np.newaxis] & (
                    np.arange(jet_mask.shape[1]) == leading_jet_pt_indices
                )
                b_tag_mask |= update_mask
                less_than_2_b_tag_jet_mask = np.sum(b_tag_mask, axis=1) < 2
                iteration += 1
                if iteration > self.max_jets:
                    raise RuntimeError(
                        "Exceeded maximum iterations while ensuring at least 2 b-tagged jets."
                    )
                    break
            jet_mask = jet_mask & b_tag_mask[:, :, np.newaxis]
        return jet_mask

    def get_neutrino_momenta(self, data_dict):
        nu_px = data_dict["non_training"][
            :,
            self.feature_index_dict["non_training"][self.neutrino_momenta_branches[0]],
        ]
        nu_py = data_dict["non_training"][
            :,
            self.feature_index_dict["non_training"][self.neutrino_momenta_branches[1]],
        ]
        nu_pz = data_dict["non_training"][
            :,
            self.feature_index_dict["non_training"][self.neutrino_momenta_branches[2]],
        ]
        anu_px = data_dict["non_training"][
            :,
            self.feature_index_dict["non_training"][self.neutrino_momenta_branches[3]],
        ]
        anu_py = data_dict["non_training"][
            :,
            self.feature_index_dict["non_training"][self.neutrino_momenta_branches[4]],
        ]
        anu_pz = data_dict["non_training"][
            :,
            self.feature_index_dict["non_training"][self.neutrino_momenta_branches[5]],
        ]
        return nu_px, nu_py, nu_pz, anu_px, anu_py, anu_pz

    def construct_neutrino_four_vectors(self, data_dict):
        nu_px, nu_py, nu_pz, anu_px, anu_py, anu_pz = self.get_neutrino_momenta(
            data_dict
        )
        nu_e = np.sqrt(nu_px**2 + nu_py**2 + nu_pz**2)
        anu_e = np.sqrt(anu_px**2 + anu_py**2 + anu_pz**2)
        return (
            np.array([nu_px, nu_py, nu_pz, nu_e]).T,
            np.array([anu_px, anu_py, anu_pz, anu_e]).T,
        )

    def get_invariant_mass(self, four_vector):
        px, py, pz, e = (
            four_vector[..., 0],
            four_vector[..., 1],
            four_vector[..., 2],
            four_vector[..., 3],
        )
        mass_squared = e**2 - (px**2 + py**2 + pz**2)
        mass_squared = np.maximum(
            mass_squared, 0
        )  # Avoid negative values due to numerical issues
        return np.sqrt(mass_squared)

    def get_four_vector(self, pt, eta, phi, e):
        px, py, pz, energy = lorentz_vector_from_pt_eta_phi_e(pt, eta, phi, e)
        return np.array([px, py, pz, energy]).transpose(
            1, 2, 0
        )  # Shape: (num_events, num_particles,4)

    def predict_indices(self, data_dict):
        """
        Predicts the indices of jets assigned to each lepton based on mass combinatorics.
        Args:
            data_dict (dict): A dictionary containing input data for the model.
        Returns:
            np.ndarray: A 3D array of shape (num_events, max_jets, max_leptons) representing the predicted indices.
        """
        leptons = data_dict["lepton"]
        jets = data_dict["jet"][
            :, :, :4
        ]  # Assuming first 4 features are pt, eta, phi, e

        lepton_four_vectors = self.get_four_vector(
            leptons[:, :, self.feature_index_dict["lepton"]["lep_pt"]],
            leptons[:, :, self.feature_index_dict["lepton"]["lep_eta"]],
            leptons[:, :, self.feature_index_dict["lepton"]["lep_phi"]],
            leptons[:, :, self.feature_index_dict["lepton"]["lep_e"]],
        )

        jet_four_vectors = self.get_four_vector(
            jets[:, :, self.feature_index_dict["jet"]["ordered_jet_pt"]],
            jets[:, :, self.feature_index_dict["jet"]["ordered_jet_eta"]],
            jets[:, :, self.feature_index_dict["jet"]["ordered_jet_phi"]],
            jets[:, :, self.feature_index_dict["jet"]["ordered_jet_e"]],
        )

        nu_four_vector, anu_four_vector = self.construct_neutrino_four_vectors(
            data_dict
        )
        viable_jets_mask = self.get_viable_jets_mask(
            data_dict
        )  # Shape: (num_events, max_jets, max_leptons)
        W_boson_candidates = lepton_four_vectors + np.stack(
            [nu_four_vector,anu_four_vector], axis=1
        )  # Shape: (num_events, max_leptons,4)
        W_boson_candidates = W_boson_candidates[:, np.newaxis, :, :]
        top_candidates = W_boson_candidates + jet_four_vectors[:, :, np.newaxis, :]
        # Shape: (num_events, max_jets, max_leptons,4)


        top_masses = self.get_invariant_mass(
            top_candidates
        )  # Shape: (num_events, max_jets, max_leptons)
        mass_differences = np.abs(
            top_masses - self.top_mass
        )  # Shape: (num_events, max_jets, max_leptons)
        mass_differences_masked = np.where(
            viable_jets_mask, mass_differences, np.inf
        )  # Shape: (num_events, max_jets, max_leptons)
        num_events = mass_differences_masked.shape[0]
        predicted_indices = np.zeros((num_events, self.max_jets, self.max_leptons))
        from scipy.optimize import linear_sum_assignment
        for e in range(num_events):
            cost_matrix = mass_differences_masked[e].T  # Shape: (max_leptons, max_jets)
            try:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
            except ValueError:
                print("Linear sum assignment failed for event", e)
                row_ind, col_ind = [], []
                continue
            predicted_indices[e, col_ind, row_ind] = 1

        return predicted_indices
