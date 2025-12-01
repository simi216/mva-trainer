"""
Data loading and preprocessing module for machine learning tasks involving jets, leptons, and MET.

This module provides classes for:
- Loading ROOT files with uproot
- Preprocessing particle physics data
- Building feature pairs for ML models
- Splitting data for training/validation

Configuration:
- LoadConfig: Used during data loading (file paths, cuts, truth labels)
- DataConfig: Describes loaded data structure (passed to ML models)
"""

import uproot
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple, Union

from .Configs import LoadConfig, DataConfig, get_load_config_from_yaml


# =============================================================================
# Raw Data Loader
# =============================================================================


class DataLoader:
    """
    Loads and pads data from ROOT files using uproot.

    Handles:
    - Loading branches from ROOT TTrees
    - Padding/clipping variable-length features
    - Converting to pandas DataFrame
    """

    def __init__(self, feature_clipping: Dict[str, int]):
        """
        Initialize the data loader.

        Args:
            feature_clipping: Dict mapping feature names to clipping values
                             (1 for scalar, >1 for variable-length features)
        """
        if not isinstance(feature_clipping, dict):
            raise ValueError("feature_clipping must be a dictionary")

        self.features = list(feature_clipping.keys())
        self.clipping = feature_clipping
        self.data = None

    def load_data(
        self, file_path: str, tree_name: str, max_events: Optional[int] = None
    ) -> None:
        """
        Load data from ROOT file.

        Args:
            file_path: Path to ROOT file
            tree_name: Name of TTree in file
            max_events: Optional limit on number of events to load
        """
        self._validate_inputs(file_path, tree_name)
        raw_data = self._load_from_root(file_path, tree_name, max_events)
        self.data = raw_data
        self._pad_and_flatten()

    def get_data(self) -> pd.DataFrame:
        """Return the loaded and processed data."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.data

    def _validate_inputs(self, file_path: str, tree_name: str) -> None:
        """Validate input parameters."""
        if not isinstance(file_path, str):
            raise ValueError("file_path must be a string")

    def _load_from_root(
        self, file_path: str, tree_name: str, max_events: Optional[int]
    ) -> ak.Array:
        """Load branches from ROOT file."""
        with uproot.open(file_path) as file:
            if tree_name not in file:
                raise ValueError(f"Tree '{tree_name}' not found in {file_path}")

            tree = file[tree_name]
            missing = [b for b in self.features if b not in tree.keys()]
            if missing:
                raise ValueError(
                    f"Missing branches in '{tree_name}': {missing}\n"
                    f"Available: {list(tree.keys())}"
                )

            kwargs = {"library": "ak"}
            if max_events is not None:
                kwargs["entry_stop"] = max_events

            data = tree.arrays(self.features, **kwargs)

        if data is None:
            raise ValueError(f"No data loaded from {file_path}")

        return data

    def _pad_and_flatten(self, padding_value: float = -999.0) -> None:
        """Pad variable-length features and convert to DataFrame."""
        data_dict = {}

        for feature in self.features:
            if feature not in self.data.fields:
                raise ValueError(f"Feature '{feature}' not found in data")

            clip_size = self.clipping[feature]

            if clip_size == 1:
                # Scalar feature
                data_dict[feature] = ak.fill_none(self.data[feature], padding_value)
            else:
                # Variable-length feature: pad and split into columns
                padded = ak.pad_none(self.data[feature], clip_size, clip=True, axis=1)
                filled = ak.fill_none(padded, padding_value)

                for i in range(clip_size):
                    data_dict[f"{feature}_{i}"] = filled[:, i]
        data = pd.DataFrame(data_dict)
        nan_filter = ~data.isna().any(axis=1)
        self.data = data[nan_filter].reset_index(drop=True)


# =============================================================================
# Feature Builders
# =============================================================================


class FeatureBuilder:
    """Builds structured feature arrays from flattened DataFrame."""

    def __init__(self, load_config: LoadConfig, data: pd.DataFrame):
        self.config = load_config
        self.data = data
        self.data_length = len(data)

    def build_all_features(self) -> Dict[str, np.ndarray]:
        """Build all feature types into structured arrays."""
        features = {}

        features["lepton"] = self._build_lepton_features()
        features["jet"] = self._build_jet_features()

        if self.config.met_features:
            features["met"] = self._build_met_features()

        if self.config.non_training_features:
            features["non_training"] = self._build_non_training_features()

        if self.config.event_weight:
            features["event_weight"] = self._build_event_weight()

        if self.config.mc_event_number:
            features["event_number"] = self._build_event_number()

        if (
            self.config.neutrino_momentum_features
            and self.config.antineutrino_momentum_features
        ):
            features["neutrino_truth"] = self._build_neutrino_truth()

        if (
            self.config.nu_flows_neutrino_momentum_features
            and self.config.nu_flows_antineutrino_momentum_features
        ):
            features["nu_flows_neutrino_truth"] = self._build_nu_flows_neutrino_truth()
        if self.config.top_truth_features and self.config.tbar_truth_features:
            features["top_truth"] = self._build_top_truth_features()
        if (
            self.config.top_lepton_truth_features
            and self.config.tbar_lepton_truth_features
        ):
            features["lepton_truth"] = self._build_lepton_truth_features()

        return features

    def _build_lepton_features(self) -> np.ndarray:
        """Build lepton feature array (n_events, NUM_LEPTONS, n_features)."""
        lepton_vars = self.config.lepton_features
        columns = [
            f"{var}_{idx}"
            for idx in range(self.config.NUM_LEPTONS)
            for var in lepton_vars
        ]

        data = self.data[columns].to_numpy()
        data = data.reshape(self.data_length, self.config.NUM_LEPTONS, -1)
        return data

    def _build_jet_features(self) -> np.ndarray:
        """Build jet feature array (n_events, max_jets, n_features)."""
        jet_vars = self.config.jet_features
        columns = [
            f"{var}_{idx}" for idx in range(self.config.max_jets) for var in jet_vars
        ]

        data = self.data[columns].to_numpy()
        data = data.reshape(self.data_length, self.config.max_jets, -1)
        return data

    def _build_met_features(self) -> np.ndarray:
        """Build MET feature array (n_events, 1, n_features)."""
        met_vars = self.config.met_features
        data = self.data[met_vars].to_numpy()
        return data.reshape(self.data_length, 1, -1)

    def _build_global_event_features(self) -> np.ndarray:
        """Build global event feature array (n_events, n_features)."""
        global_event_vars = self.config.global_event_features
        data = self.data[global_event_vars].to_numpy()
        return data.reshape(self.data_length, -1)

    def _build_non_training_features(self) -> np.ndarray:
        """Build non-training feature array (n_events, n_features)."""
        return self.data[self.config.non_training_features].to_numpy()

    def _build_event_weight(self) -> np.ndarray:
        """Build event weight array (n_events,)."""
        return self.data[self.config.event_weight].to_numpy()

    def _build_event_number(self) -> np.ndarray:
        """Build event number array (n_events,)."""
        return self.data[self.config.mc_event_number].to_numpy()

    def _build_neutrino_truth(self) -> np.ndarray:
        """Build regression target array (n_events, NUM_LEPTONS, n_targets)."""
        neutrino_vars = self.config.neutrino_momentum_features
        antineutrino_vars = self.config.antineutrino_momentum_features
        columns = neutrino_vars + antineutrino_vars
        data = self.data[columns].to_numpy()
        data = data.reshape(self.data_length, self.config.NUM_LEPTONS, -1)  # n_targets
        return data

    def _build_nu_flows_neutrino_truth(self) -> np.ndarray:
        """Build NuFlows regression target array (n_events, NUM_LEPTONS, n_targets)."""
        nu_flows_vars = (
            self.config.nu_flows_neutrino_momentum_features
            + self.config.nu_flows_antineutrino_momentum_features
        )

        data = self.data[nu_flows_vars].to_numpy()
        data = data.reshape(self.data_length, self.config.NUM_LEPTONS, -1)  # n_targets
        return data

    def _build_top_truth_features(self) -> np.ndarray:
        """Build top quark truth feature array (n_events, n_features)."""
        top_truth_vars = (
            self.config.top_truth_features + self.config.tbar_truth_features
        )
        data = (
            self.data[top_truth_vars]
            .to_numpy()
            .reshape(self.data_length, self.config.NUM_LEPTONS, -1)
        )
        return data

    def _build_lepton_truth_features(self) -> np.ndarray:
        """Build lepton truth feature array (n_events, n_features)."""
        lepton_truth_vars = (
            self.config.top_lepton_truth_features
            + self.config.tbar_lepton_truth_features
        )
        data = (
            self.data[lepton_truth_vars]
            .to_numpy()
            .reshape(self.data_length, self.config.NUM_LEPTONS, -1)
        )
        return data


class LabelBuilder:
    """Builds training labels from truth information."""

    def __init__(
        self,
        load_config: LoadConfig,
        data: Union[pd.DataFrame, tuple[np.ndarray, np.ndarray]],
    ):
        self.config = load_config
        self.data = data

    def build_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build labels from jet and lepton truth information.

        Returns:
            pair_truth: Binary pairing labels (n_events, max_jets, NUM_LEPTONS)
            reco_mask: Boolean mask for successfully reconstructed events
        """
        jet_truth = self._extract_jet_truth()
        lepton_truth = self._extract_lepton_truth()
        reco_mask = self._get_reconstruction_mask(jet_truth)

        # Filter truth by reconstruction mask
        jet_truth = jet_truth[reco_mask]
        lepton_truth = lepton_truth[reco_mask]

        pair_truth = self._build_pair_truth_tensor(jet_truth, lepton_truth)

        return pair_truth, reco_mask

    def _extract_jet_truth(self) -> np.ndarray:
        """Extract jet truth indices."""
        if isinstance(self.data, pd.DataFrame):
            cols = [f"{self.config.jet_truth_label}_{i}" for i in [0, 3]]
            return self.data[cols].to_numpy()
        else:
            return self.data[0][:, [0, 3]]

    def _extract_lepton_truth(self) -> np.ndarray:
        """Extract lepton truth indices."""
        if isinstance(self.data, pd.DataFrame):
            cols = [f"{self.config.lepton_truth_label}_{i}" for i in [0, 1]]
            return self.data[cols].to_numpy()
        else:
            return self.data[1]

    def _get_reconstruction_mask(self, jet_truth: np.ndarray) -> np.ndarray:
        """Get mask for events with valid reconstruction."""
        valid_indices = (jet_truth != -1).all(axis=1)
        in_range = (jet_truth < self.config.max_jets).all(axis=1)
        return valid_indices & in_range

    def _build_pair_truth_tensor(
        self, jet_truth: np.ndarray, lepton_truth: np.ndarray
    ) -> np.ndarray:
        """Build binary tensor indicating correct jet-lepton pairs."""
        n_events = len(jet_truth)
        pair_truth = np.zeros(
            (n_events, self.config.max_jets, self.config.NUM_LEPTONS), dtype=np.float32
        )

        for evt_idx in range(n_events):
            for lep_idx in range(self.config.NUM_LEPTONS):
                for jet_idx in range(self.config.max_jets):
                    # Check both possible pairings
                    if (
                        jet_truth[evt_idx, 0] == jet_idx
                        and lepton_truth[evt_idx, 0] == lep_idx
                    ):
                        pair_truth[evt_idx, jet_idx, lep_idx] = 1
                    elif (
                        jet_truth[evt_idx, 1] == jet_idx
                        and lepton_truth[evt_idx, 1] == lep_idx
                    ):
                        pair_truth[evt_idx, jet_idx, lep_idx] = 1

        return pair_truth


# =============================================================================
# Main Preprocessor
# =============================================================================


class DataPreprocessor:
    """
    Main class for loading and preprocessing particle physics data.

    Handles:
    - Loading ROOT files
    - Applying cuts
    - Building feature arrays
    - Creating labels
    - Splitting data for training
    - Data normalization

    After loading, provides a DataConfig that describes the loaded data
    structure for downstream components (e.g., ML models).
    """

    def __init__(self, load_config: LoadConfig):
        """
        Initialize preprocessor with loading configuration.

        Args:
            load_config: Configuration for data loading
        """
        self.load_config = load_config
        self.data_config: Optional[DataConfig] = None

        self.data = None
        self.data_length = None
        self.feature_data = None
        self.cut_dict = {}
        self.data_normalisation_factors = {}

    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------

    def load_data(
        self,
        file_path: str,
        tree_name: str,
        max_events: Optional[int] = None,
        cut_neg_weights: bool = True,
    ) -> DataConfig:
        """
        Load and preprocess data from ROOT file.

        After loading, creates a DataConfig describing the loaded data structure.

        Args:
            file_path: Path to ROOT file
            tree_name: Name of TTree
            max_events: Optional limit on events to load
            cut_neg_weights: Whether to remove negative event weights
        """
        if self.data is not None:
            raise ValueError("Data already loaded. Create new instance for new data.")

        # Load raw data
        feature_clipping = self.load_config.get_feature_clipping_dict()
        loader = DataLoader(feature_clipping)
        loader.load_data(file_path, tree_name, max_events)
        self.data = loader.get_data()

        # Apply preprocessing
        self._filter_negative_weights(cut_neg_weights)
        self.data_length = len(self.data)

        # Build features and labels
        self._build_features_and_labels()

        # Create DataConfig for downstream use
        self.data_config = self.load_config.to_data_config()
        return self.data_config

    def get_data_config(self) -> DataConfig:
        """
        Get the DataConfig describing loaded data structure.

        This should be passed to downstream components like ML models.

        Returns:
            DataConfig instance

        Raises:
            ValueError: If data has not been loaded yet
        """
        if self.data_config is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.data_config

    def _filter_negative_weights(self, cut_neg_weights: bool) -> None:
        """Remove events with negative weights if requested."""
        if self.load_config.event_weight and cut_neg_weights:
            n_before = len(self.data)
            self.data = self.data[self.data[self.load_config.event_weight] >= 0]
            n_after = len(self.data)
            if n_before != n_after:
                print(f"Removed {n_before - n_after} events with negative weights")

    def _build_features_and_labels(self) -> None:
        """Build structured feature arrays and labels."""
        # Build features
        feature_builder = FeatureBuilder(self.load_config, self.data)
        self.feature_data = feature_builder.build_all_features()

        # Build labels and apply reconstruction mask
        label_builder = LabelBuilder(self.load_config, self.data)
        pair_truth, reco_mask = label_builder.build_labels()

        # Apply reconstruction mask to all features
        for key in self.feature_data:
            if self.feature_data[key] is not None:
                self.feature_data[key] = self.feature_data[key][reco_mask]

        self.data_length = len(pair_truth)

        self.feature_data["assignment_labels"] = pair_truth

        # Clear DataFrame to save memory
        self.data = None

    # -------------------------------------------------------------------------
    # Custom Features
    # -------------------------------------------------------------------------

    def add_custom_feature(self, function: callable, name: str) -> None:
        """
        Add a custom feature computed from existing features.

        Args:
            function: Function that takes feature_data dict and returns new feature
            name: Name for the new feature
        """
        if self.feature_data is None:
            raise ValueError("Feature data not loaded. Call load_data() first.")
        if self.data_config is None:
            raise ValueError("DataConfig not available. Call load_data() first.")

        new_data = function(self.feature_data, self.data_config)

        if len(new_data) != self.data_length:
            raise ValueError(
                f"Custom feature must have {self.data_length} events, "
                f"got {len(new_data)}"
            )

        if new_data.ndim == 1:
            new_data = new_data.reshape(-1, 1)

        # Add to feature data
        if "custom" not in self.feature_data:
            self.feature_data["custom"] = new_data
            feature_idx = 0
        else:
            feature_idx = self.feature_data["custom"].shape[1]
            self.feature_data["custom"] = np.concatenate(
                (self.feature_data["custom"], new_data), axis=1
            )
        self.data_config.add_custom_feature(name, feature_idx)

    # -------------------------------------------------------------------------
    # Data Access
    # -------------------------------------------------------------------------

    def get_feature_data(self, data_type: str, feature_name: str) -> np.ndarray:
        """
        Get specific feature data.

        Args:
            data_type: Type of feature ('jet', 'lepton', 'met', 'non_training', etc.)
            feature_name: Name of specific feature

        Returns:
            Feature array (shape depends on data_type)
        """
        if self.feature_data is None:
            raise ValueError("Feature data not prepared. Call load_data() first.")

        if self.data_config is None:
            raise ValueError("DataConfig not available. Call load_data() first.")

        # Get feature index from DataConfig
        feature_idx = self.data_config.get_feature_index(data_type, feature_name)

        if data_type in ["jet", "lepton", "met"]:
            return self.feature_data[data_type][:, :, feature_idx].copy()
        elif data_type in ["non_training", "custom"]:
            return self.feature_data[data_type][:, feature_idx].copy()
        elif data_type == "event_weight":
            return self.get_event_weight().copy()
        elif data_type == "event_number":
            return self.get_event_number().copy()
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    def get_all_feature_data(self, feature_type: str) -> np.ndarray:
        """Get all features of a given type."""
        if self.feature_data is None:
            raise ValueError("Feature data not prepared. Call load_data() first.")

        if feature_type not in self.feature_data:
            raise ValueError(f"Feature type '{feature_type}' not found")

        return self.feature_data[feature_type].copy()

    def get_event_weight(self, cut_neg_weights: bool = False) -> np.ndarray:
        """
        Get normalized event weights.

        Args:
            cut_neg_weights: Whether to exclude negative weights

        Returns:
            Normalized event weights
        """
        if self.feature_data is None:
            raise ValueError("Feature data not prepared. Call load_data() first.")

        if self.load_config.event_weight is None:
            raise ValueError("Event weight not configured")

        weights = self.feature_data["event_weight"]

        if cut_neg_weights:
            weights = weights[weights >= 0]

        # Normalize to sum to 1
        weights = weights / np.sum(weights)

        return weights

    def get_event_number(self) -> np.ndarray:
        """
        Get event numbers.

        Returns:
            Event numbers
        """
        if self.feature_data is None:
            raise ValueError("Feature data not prepared. Call load_data() first.")

        if self.load_config.mc_event_number is None:
            raise ValueError("Event number not configured")

        event_numbers = self.feature_data["event_number"]

        return event_numbers

    # -------------------------------------------------------------------------
    # Data Splitting
    # -------------------------------------------------------------------------

    def split_data(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """
        Split data into training and testing sets.

        Args:
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility

        Returns:
            X_train, y_train, X_test, y_test
        """
        if self.feature_data is None:
            raise ValueError("Data not prepared. Call load_data() first.")

        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = {}, {}, None, None

        for key, data in self.feature_data.items():
            if data is not None:
                X_train[key], X_test[key] = train_test_split(
                    data, test_size=test_size, random_state=random_state
                )
        y_train = {
            "assignment_labels": X_train["assignment_labels"],
            "neutrino_truth": X_train.get("neutrino_truth", None),
        }
        y_test = {
            "assignment_labels": X_test["assignment_labels"],
            "neutrino_truth": X_test.get("neutrino_truth", None),
        }

        return X_train, y_train, X_test, y_test

    def get_data(self):
        X, Y = {}, {}
        for key, data in self.feature_data.items():
            if data is not None:
                X[key] = data
        Y = {
            "assignment_labels": X["assignment_labels"],
            "neutrino_truth": X.get("neutrino_truth", None),
        }
        return X, Y

    def create_k_folds(
        self, n_folds: int = 5, n_splits: int = 1, random_state: int = 42
    ) -> List[
        Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray], np.ndarray]
    ]:
        """
        Create k-fold cross-validation splits.

        Args:
            n_folds: Number of folds
            n_splits: Number of independent splits
            random_state: Random seed

        Returns:
            List of (X_train, y_train, X_test, y_test) tuples
        """
        if self.feature_data is None:
            raise ValueError("Data not prepared. Call load_data() first.")

        if random_state is not None:
            np.random.seed(random_state)

        # Shuffle data once
        indices = np.arange(self.data_length)
        np.random.shuffle(indices)

        for key in self.feature_data:
            if self.feature_data[key] is not None:
                self.feature_data[key] = self.feature_data[key][indices]

        # Create folds
        split_size = self.data_length // n_splits
        folds = []

        for split_idx in range(n_splits):
            start = split_idx * split_size
            end = (split_idx + 1) * split_size

            split_len = end - start
            fold_size = split_len // n_folds

            for fold_idx in range(n_folds):
                test_start = start + fold_idx * fold_size
                test_end = test_start + fold_size if fold_idx < n_folds - 1 else end

                # Training data: everything except test fold
                train_indices = np.concatenate(
                    [np.arange(start, test_start), np.arange(test_end, end)]
                )
                test_indices = np.arange(test_start, test_end)

                X_train = {
                    k: self.feature_data[k][train_indices] for k in self.feature_data
                }
                y_train = {
                    "assignment_labels": X_train["assignment_labels"],
                    "neutrino_truth": X_train.get("neutrino_truth", None),
                }
                y_test = {
                    "assignment_labels": self.feature_data["assignment_labels"][
                        test_indices
                    ],
                    "neutrino_truth": self.feature_data.get("neutrino_truth", None)[
                        test_indices
                    ],
                }

                X_test = {
                    k: self.feature_data[k][test_indices] for k in self.feature_data
                }

                folds.append((X_train, y_train, X_test, y_test))

        return folds

    def split_even_odd(
        self,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """
        Split data into even and odd event number sets.

        Returns:
            X_even, y_even, X_odd, y_odd
        """
        if self.feature_data is None:
            raise ValueError("Data not prepared. Call load_data() first.")

        if "event_number" not in self.feature_data:
            raise ValueError("Event number feature not available for splitting.")

        event_numbers = self.feature_data["event_number"]
        even_mask = (event_numbers % 2) == 0
        odd_mask = ~even_mask

        X_even = {k: v[even_mask] for k, v in self.feature_data.items()}
        y_even = {
            "assignment_labels": X_even["assignment_labels"],
            "neutrino_truth": X_even.get("neutrino_truth", None),
        }

        X_odd = {k: v[odd_mask] for k, v in self.feature_data.items()}
        y_odd = {
            "assignment_labels": X_odd["assignment_labels"],
            "neutrino_truth": X_odd.get("neutrino_truth", None),
        }

        return X_even, y_even, X_odd, y_odd

    def load_from_npz(
        self, npz_path: str, max_events: Optional[int] = None
    ) -> DataConfig:
        """
        Load preprocessed data from NPZ file.

        Args:
            npz_path: Path to NPZ file
        """
        loaded = np.load(npz_path)

        if max_events is not None:
            loaded_array = {}
            for key in loaded.files:
                loaded_array[key] = loaded[key][:max_events]
            loaded = loaded_array

        if loaded is None:
            raise ValueError(f"Could not read file {npz_path}")

        self.feature_data = {}

        if self.load_config.jet_features:
            self.feature_data["jet"] = np.array(
                [loaded[jet_key] for jet_key in self.load_config.jet_features]
            ).transpose(1, 2, 0)[:, : self.load_config.max_jets, :]

        if self.load_config.lepton_features:
            self.feature_data["lepton"] = np.array(
                [loaded[lep_key] for lep_key in self.load_config.lepton_features]
            ).transpose(1, 2, 0)

        if self.load_config.global_event_features:
            self.feature_data["global_event"] = np.array(
                [loaded[ge_key] for ge_key in self.load_config.global_event_features]
            ).transpose(1, 0)

        if self.load_config.met_features:
            self.feature_data["met"] = np.array(
                [loaded[met_key] for met_key in self.load_config.met_features]
            ).transpose(1, 0)[:, np.newaxis, :]

        if self.load_config.non_training_features:
            self.feature_data["non_training"] = np.array(
                [loaded[nt_key] for nt_key in self.load_config.non_training_features]
            ).transpose(1, 0)

        if self.load_config.event_weight:
            self.feature_data["event_weight"] = loaded[self.load_config.event_weight]

        if self.load_config.mc_event_number:
            self.feature_data["event_number"] = loaded[self.load_config.mc_event_number]

        if (
            self.load_config.neutrino_momentum_features
            and self.load_config.antineutrino_momentum_features
        ):
            data_length = len(loaded[self.load_config.neutrino_momentum_features[0]])
            self.feature_data["neutrino_truth"] = (
                np.array(
                    [
                        loaded[nu_key]
                        for nu_key in self.load_config.neutrino_momentum_features
                        + self.load_config.antineutrino_momentum_features
                    ]
                )
                .transpose(1, 0)
                .reshape(data_length, self.load_config.NUM_LEPTONS, -1)
            )

        if (
            self.load_config.nu_flows_neutrino_momentum_features
            and self.load_config.nu_flows_antineutrino_momentum_features
        ):
            data_length = len(
                loaded[self.load_config.nu_flows_neutrino_momentum_features[0]]
            )
            self.feature_data["nu_flows_neutrino_truth"] = (
                np.array(
                    [
                        loaded[nu_key]
                        for nu_key in self.load_config.nu_flows_neutrino_momentum_features
                        + self.load_config.nu_flows_antineutrino_momentum_features
                    ]
                )
                .transpose(1, 0)
                .reshape(data_length, self.load_config.NUM_LEPTONS, -1)
            )
        if self.load_config.top_truth_features and self.load_config.tbar_truth_features:
            data_length = len(loaded[self.load_config.top_truth_features[0]])
            self.feature_data["top_truth"] = (
                np.array(
                    [
                        loaded[top_key]
                        for top_key in self.load_config.top_truth_features
                        + self.load_config.tbar_truth_features
                    ]
                )
                .transpose(1, 0)
                .reshape(data_length, self.load_config.NUM_LEPTONS, -1)
            )

        if (
            self.load_config.top_lepton_truth_features
            and self.load_config.tbar_lepton_truth_features
        ):
            data_length = len(loaded[self.load_config.top_lepton_truth_features[0]])
            self.feature_data["lepton_truth"] = (
                np.array(
                    [
                        loaded[lep_key]
                        for lep_key in self.load_config.top_lepton_truth_features
                        + self.load_config.tbar_lepton_truth_features
                    ]
                )
                .transpose(1, 0)
                .reshape(data_length, self.load_config.NUM_LEPTONS, -1)
            )

        # Build labels and apply reconstruction mask
        label_builder = LabelBuilder(
            self.load_config,
            (
                loaded[self.load_config.jet_truth_label],
                loaded[self.load_config.lepton_truth_label],
            ),
        )
        assignment_labels, reco_mask = label_builder.build_labels()

        # Apply reconstruction mask to all features
        for key in self.feature_data:
            if self.feature_data[key] is not None:
                self.feature_data[key] = self.feature_data[key][reco_mask]

        self.feature_data["assignment_labels"] = assignment_labels

        # Remove events with NaNs in NuFlows targets if present
        if "nu_flows_neutrino_truth" in self.feature_data:
            nu_flows_nan_mask = ~np.isnan(
                self.feature_data["nu_flows_neutrino_truth"]
            ).any(axis=(1, 2))
            for key in self.feature_data:
                if self.feature_data[key] is not None:
                    self.feature_data[key] = self.feature_data[key][nu_flows_nan_mask]

        self.data_length = len(self.feature_data["assignment_labels"])

        # Create DataConfig for downstream use
        self.data_config = self.load_config.to_data_config()
        return self.data_config


def combine_datasets(
    data_list: List[DataPreprocessor],
) -> DataPreprocessor:
    """
    Combine multiple DataPreprocessor instances into one.

    Args:
        data_list: List of DataPreprocessor instances to combine

    Returns:
        Combined DataPreprocessor instance
    """
    if not data_list:
        raise ValueError("data_list must contain at least one DataPreprocessor")

    combined = DataPreprocessor(data_list[0].load_config)

    combined.feature_data = {}
    for key in data_list[0].feature_data.keys():
        combined.feature_data[key] = np.concatenate(
            [dp.feature_data[key] for dp in data_list], axis=0
        )

    combined.data_length = sum(dp.data_length for dp in data_list)
    combined.data_config = data_list[0].data_config

    return combined


def combine_train_datasets(
    X: List[dict[str : np.ndarray]],
    y: List[dict[str : np.ndarray]],
    weights: Optional[List[float]] = None,
) -> dict[str : np.ndarray]:
    """
    Combine multiple training datasets into one.

    Args:
        X: List of feature dictionaries
        y: List of label dictionaries
    Returns:
        Combined feature dictionary
        combined_X = {}
        combined_y = {}
    """
    combined_X = {}
    combined_y = {}

    for key in X[0].keys():
        combined_X[key] = np.concatenate([x[key] for x in X], axis=0)

    for key in y[0].keys():
        combined_y[key] = np.concatenate([label[key] for label in y], axis=0)

    if weights is not None:
        # Apply weights to event weights if present
        if "event_weight" in combined_X:
            event_weights = []
            for i, x in enumerate(X):
                n_events = x["event_weight"].shape[0]
                event_weights.append(
                    x["event_weight"] / np.mean(x["event_weight"]) * weights[i]
                )
            combined_X["event_weight"] = np.concatenate(event_weights, axis=0)

    permutation = np.random.permutation(combined_X[list(combined_X.keys())[0]].shape[0])
    for key in combined_X.keys():
        combined_X[key] = combined_X[key][permutation]
    for key in combined_y.keys():
        combined_y[key] = combined_y[key][permutation]

    return combined_X, combined_y
