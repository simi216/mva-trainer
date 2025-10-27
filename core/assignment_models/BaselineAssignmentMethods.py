from core.reconstruction import EventReconstructorBase
from core.DataLoader import DataConfig
import numpy as np
from core.utils.tools import four_vector_from_pt_eta_phi_e, compute_mass_from_four_vector


class BaselineAssigner(EventReconstructorBase):
    def __init__(self, config: DataConfig, name="baseline_assigner", mode = "min"):
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
        raise NotImplementedError("This method should be implemented with actual logic.")

    def get_viable_jets_mask(self, data_dict):
        """
        Computes a mask indicating viable jets for each lepton.
        This is a placeholder implementation and should be replaced with actual logic.

        Args:
            data_dict (dict): A dictionary containing input data for the model.
        Returns:
            np.ndarray: A 3D boolean array of shape (num_events, max_jets, max_leptons) indicating viable jets.
        """
        padding_value =self.config.padding_value
        jet_mask = (data_dict["jet"] != padding_value).all(axis=-1)  # Shape: (num_events, max_jets)
        return np.repeat(jet_mask[:, :,np.newaxis], self.max_leptons, axis=2)  # Shape: (num_events, max_jets, max_leptons)

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
        predicted_indices = np.zeros((num_events, self.max_jets,self.max_leptons))
        for i in range(num_events):
            for j in range(self.max_leptons):
                valid_indices = np.where(viable_jets_mask[i,:, j])[0]
                if valid_indices.size > 0:
                    if self.mode == "min":
                        best_jet_index = valid_indices[np.argmin(comparison_feature[i, valid_indices,j])]
                    else:  # self.mode == "max"
                        best_jet_index = valid_indices[np.argmax(comparison_feature[i, valid_indices,j])]
                    predicted_indices[i,best_jet_index, j] = 1
                    viable_jets_mask[i,best_jet_index, :] = False  # Exclude this jet for other leptons
        return predicted_indices


class DeltaRAssigner(BaselineAssigner):
    def __init__(self, config: DataConfig, name="delta_r_assigner", mode = "min", b_tag_threshold = 2):
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
        if lepton_eta is None or lepton_phi is None or jet_eta is None or jet_phi is None:
            raise ValueError("Eta and Phi features must be present in both leptons and jets.")
        delta_eta = jet_eta[:, :, np.newaxis] - lepton_eta[:, np.newaxis, :]
        delta_phi = jet_phi[:, :, np.newaxis] - lepton_phi[:, np.newaxis, :]
        delta_phi = (delta_phi + np.pi) % (2 * np.pi) - np.pi  # Wrap-around
        delta_r = np.sqrt(delta_eta**2 + delta_phi**2)
        return delta_r
    
    def get_viable_jets_mask(self, data_dict):
        """
        Computes a mask indicating viable jets for each lepton based on the padding value.

        Args:
            data_dict (dict): A dictionary containing input data for the model.
        Returns:
            np.ndarray: A 3D boolean array of shape (num_events, max_leptons, max_jets) indicating viable jets.
        """
        jet_mask = super().get_viable_jets_mask(data_dict)
        jet_b_tag = None
        for feature in self.jet_features:
            if "btag" in feature.lower() or "b_tag" in feature.lower():
                jet_b_tag = data_dict["jet"][:, :, self.feature_index_dict["jet"][feature]]
        if jet_b_tag is not None:
            b_tag_mask = jet_b_tag >= 2 & jet_mask[:, :,0]
            less_than_2_btagged_jets = np.sum(b_tag_mask, axis=1) < 2
            b_tag_mask[less_than_2_btagged_jets, :] = True
            jet_mask = jet_mask & b_tag_mask[:, :, np.newaxis]
        return jet_mask
    
 
class LeptonJetMassAssigner(BaselineAssigner):
    def __init__(self, config: DataConfig, name="lepton_jet_mass_assigner", mode = "min"):
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
        lepton_energy = leptons[:, :, self.feature_index_dict["lepton"]["lep_e"]][:, np.newaxis, :]
        lepton_pt = leptons[:, :, self.feature_index_dict["lepton"]["lep_pt"]][:, np.newaxis, :]
        lepton_eta = leptons[:, :, self.feature_index_dict["lepton"]["lep_eta"]][:, np.newaxis, :]
        lepton_phi = leptons[:, :, self.feature_index_dict["lepton"]["lep_phi"]][:, np.newaxis, :]
        jet_energy = jets[:, :, self.feature_index_dict["jet"]["ordered_jet_e"]][:, :, np.newaxis]
        jet_pt = jets[:, :, self.feature_index_dict["jet"]["ordered_jet_pt"]][:, :, np.newaxis]
        jet_eta = jets[:, :, self.feature_index_dict["jet"]["ordered_jet_eta"]][:, :, np.newaxis]
        jet_phi = jets[:, :, self.feature_index_dict["jet"]["ordered_jet_phi"]][:, :, np.newaxis]
        lep_px, lep_py, lep_pz, lep_e = four_vector_from_pt_eta_phi_e(
            lepton_pt, lepton_eta, lepton_phi, lepton_energy
        )
        jet_px, jet_py, jet_pz, jet_e = four_vector_from_pt_eta_phi_e(
            jet_pt, jet_eta, jet_phi, jet_energy
        )
        combined_px = lep_px + jet_px
        combined_py = lep_py + jet_py
        combined_pz = lep_pz + jet_pz
        combined_e = lep_e + jet_e
        invariant_mass = compute_mass_from_four_vector(
            combined_px, combined_py, combined_pz, combined_e
        )
        return invariant_mass
    
    def get_viable_jets_mask(self, data_dict):
        """
        Computes a mask indicating viable jets for each lepton based on the padding value.

        Args:
            data_dict (dict): A dictionary containing input data for the model.
        Returns:
            np.ndarray: A 3D boolean array of shape (num_events, max_leptons, max_jets) indicating viable jets.
        """
        jet_mask = super().get_viable_jets_mask(data_dict)
        jet_b_tag = None
        for feature in self.jet_features:
            if "btag" in feature.lower() or "b_tag" in feature.lower():
                jet_b_tag = data_dict["jet"][:, :, self.feature_index_dict["jet"][feature]]
        if jet_b_tag is not None:
            b_tag_mask = jet_b_tag >= 2 & jet_mask[:, :,0]
            less_than_2_btagged_jets = np.sum(b_tag_mask, axis=1) < 2
            b_tag_mask[less_than_2_btagged_jets, :] = True
            jet_mask = jet_mask & b_tag_mask[:, :, np.newaxis]
        return jet_mask
