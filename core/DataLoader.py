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
from typing import Optional, Dict, List, Tuple

from .Configs import LoadConfig, DataConfig


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
        self, 
        file_path: str, 
        tree_name: str, 
        max_events: Optional[int] = None
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
        self, 
        file_path: str, 
        tree_name: str, 
        max_events: Optional[int]
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
                padded = ak.pad_none(
                    self.data[feature], clip_size, clip=True, axis=1
                )
                filled = ak.fill_none(padded, padding_value)
                
                for i in range(clip_size):
                    data_dict[f"{feature}_{i}"] = filled[:, i]
        
        self.data = pd.DataFrame(data_dict)


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
        
        if self.config.regression_targets:
            features["regression_targets"] = self._build_regression_targets()
        
        return features
    
    def _build_lepton_features(self) -> np.ndarray:
        """Build lepton feature array (n_events, max_leptons, n_features)."""
        lepton_vars = self.config.lepton_features
        columns = [
            f"{var}_{idx}"
            for var in lepton_vars
            for idx in range(self.config.max_leptons)
        ]
        
        data = self.data[columns].to_numpy()
        data = data.reshape(self.data_length, -1, self.config.max_leptons)
        return data.transpose((0, 2, 1))
    
    def _build_jet_features(self) -> np.ndarray:
        """Build jet feature array (n_events, max_jets, n_features)."""
        jet_vars = self.config.jet_features
        columns = [
            f"{var}_{idx}"
            for var in jet_vars
            for idx in range(self.config.max_jets)
        ]
        
        data = self.data[columns].to_numpy()
        data = data.reshape(self.data_length, -1, self.config.max_jets)
        return data.transpose((0, 2, 1))
    
    def _build_met_features(self) -> np.ndarray:
        """Build MET feature array (n_events, 1, n_features)."""
        met_vars = self.config.met_features
        data = self.data[met_vars].to_numpy()
        return data.reshape(self.data_length, 1, -1)
    
    def _build_non_training_features(self) -> np.ndarray:
        """Build non-training feature array (n_events, n_features)."""
        return self.data[self.config.non_training_features].to_numpy()
    
    def _build_event_weight(self) -> np.ndarray:
        """Build event weight array (n_events,)."""
        return self.data[self.config.event_weight].to_numpy()
    
    def _build_regression_targets(self) -> np.ndarray:
        """Build regression target array (n_events, n_targets)."""
        return self.data[self.config.regression_targets].to_numpy()


class LabelBuilder:
    """Builds training labels from truth information."""
    
    def __init__(self, load_config: LoadConfig, data: pd.DataFrame):
        self.config = load_config
        self.data = data
    
    def build_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build labels from jet and lepton truth information.
        
        Returns:
            pair_truth: Binary pairing labels (n_events, max_jets, max_leptons)
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
        cols = [f"{self.config.jet_truth_label}_{i}" for i in [0, 3]]
        return self.data[cols].to_numpy()
    
    def _extract_lepton_truth(self) -> np.ndarray:
        """Extract lepton truth indices."""
        cols = [f"{self.config.lepton_truth_label}_{i}" for i in [0, 1]]
        return self.data[cols].to_numpy()
    
    def _get_reconstruction_mask(self, jet_truth: np.ndarray) -> np.ndarray:
        """Get mask for events with valid reconstruction."""
        valid_indices = (jet_truth != -1).all(axis=1)
        in_range = (jet_truth < self.config.max_jets).all(axis=1)
        return valid_indices & in_range
    
    def _build_pair_truth_tensor(
        self, 
        jet_truth: np.ndarray, 
        lepton_truth: np.ndarray
    ) -> np.ndarray:
        """Build binary tensor indicating correct jet-lepton pairs."""
        n_events = len(jet_truth)
        pair_truth = np.zeros(
            (n_events, self.config.max_jets, self.config.max_leptons),
            dtype=np.float32
        )
        
        for evt_idx in range(n_events):
            for lep_idx in range(self.config.max_leptons):
                for jet_idx in range(self.config.max_jets):
                    # Check both possible pairings
                    if (jet_truth[evt_idx, 0] == jet_idx and 
                        lepton_truth[evt_idx, 0] == lep_idx):
                        pair_truth[evt_idx, jet_idx, lep_idx] = 1
                    elif (jet_truth[evt_idx, 1] == jet_idx and 
                          lepton_truth[evt_idx, 1] == lep_idx):
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
        self.labels = None
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
        cut_neg_weights: bool = True
    ) -> None:
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
        self._apply_cuts()
        self._filter_negative_weights(cut_neg_weights)
        self.data_length = len(self.data)
        
        # Build features and labels
        self._build_features_and_labels()
        
        # Create DataConfig for downstream use
        self.data_config = self.load_config.to_data_config()

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

    def _apply_cuts(self) -> None:
        """Apply all registered feature cuts."""
        for cut_feature, (cut_low, cut_high) in self.cut_dict.items():
            if cut_feature not in self.data.columns:
                raise ValueError(f"Cut feature '{cut_feature}' not found in data")
            
            print(f"Applying cut on '{cut_feature}': [{cut_low}, {cut_high}]")
            print(f"  Range before: [{self.data[cut_feature].min():.2f}, "
                  f"{self.data[cut_feature].max():.2f}]")
            
            if cut_low is not None:
                self.data = self.data[self.data[cut_feature] >= cut_low]
            if cut_high is not None:
                self.data = self.data[self.data[cut_feature] <= cut_high]
            
            print(f"  Events remaining: {len(self.data)}")

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
        
        # Set labels
        if self.load_config.regression_targets:
            self.labels = self.feature_data["regression_targets"]
        else:
            self.labels = pair_truth
        
        self.feature_data["labels"] = pair_truth
        
        # Clear DataFrame to save memory
        self.data = None

    # -------------------------------------------------------------------------
    # Cut Management
    # -------------------------------------------------------------------------

    def add_cut(
        self, 
        cut_feature: str, 
        cut_low: Optional[float] = None, 
        cut_high: Optional[float] = None
    ) -> None:
        """
        Register a cut to be applied on a feature.
        
        Args:
            cut_feature: Name of feature to cut on
            cut_low: Lower bound (inclusive)
            cut_high: Upper bound (inclusive)
        """
        if cut_low is None and cut_high is None:
            raise ValueError("Must specify at least one of cut_low or cut_high")
        
        if cut_low is not None and cut_high is not None and cut_low >= cut_high:
            raise ValueError("cut_low must be less than cut_high")
        
        self.cut_dict[cut_feature] = (cut_low, cut_high)

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
        
        new_data = function(self.feature_data)
        
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
        
        # Update DataConfig if available
        if self.data_config is not None:
            self.data_config.add_custom_feature(name, feature_idx)

    # -------------------------------------------------------------------------
    # Data Access
    # -------------------------------------------------------------------------

    def get_labels(self) -> np.ndarray:
        """Get training labels."""
        if self.labels is None:
            raise ValueError("Labels not prepared. Call load_data() first.")
        return self.labels

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
            return self.get_event_weight()
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

    # -------------------------------------------------------------------------
    # Data Splitting
    # -------------------------------------------------------------------------

    def split_data(
        self, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        """
        Split data into training and testing sets.
        
        Args:
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, y_train, X_test, y_test
        """
        if self.feature_data is None or self.labels is None:
            raise ValueError("Data not prepared. Call load_data() first.")
        
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = {}, {}, None, None
        
        for key, data in self.feature_data.items():
            if data is not None:
                X_train[key], X_test[key], y_train, y_test = train_test_split(
                    data, self.labels,
                    test_size=test_size,
                    random_state=random_state
                )
        
        return X_train, y_train, X_test, y_test

    def create_k_folds(
        self,
        n_folds: int = 5,
        n_splits: int = 1,
        random_state: int = 42
    ) -> List[Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, np.ndarray], np.ndarray]]:
        """
        Create k-fold cross-validation splits.
        
        Args:
            n_folds: Number of folds
            n_splits: Number of independent splits
            random_state: Random seed
            
        Returns:
            List of (X_train, y_train, X_test, y_test) tuples
        """
        if self.feature_data is None or self.labels is None:
            raise ValueError("Data not prepared. Call load_data() first.")
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Shuffle data once
        indices = np.arange(self.data_length)
        np.random.shuffle(indices)
        
        for key in self.feature_data:
            if self.feature_data[key] is not None:
                self.feature_data[key] = self.feature_data[key][indices]
        self.labels = self.labels[indices]
        
        # Create folds
        split_size = self.data_length // n_splits
        folds = []
        
        for split_idx in range(n_splits):
            start = split_idx * split_size
            end = (split_idx + 1) * split_size if split_idx < n_splits - 1 else self.data_length
            
            split_len = end - start
            fold_size = split_len // n_folds
            
            for fold_idx in range(n_folds):
                test_start = start + fold_idx * fold_size
                test_end = test_start + fold_size if fold_idx < n_folds - 1 else end
                
                # Training data: everything except test fold
                train_indices = np.concatenate([
                    np.arange(start, test_start),
                    np.arange(test_end, end)
                ])
                test_indices = np.arange(test_start, test_end)
                
                X_train = {
                    k: self.feature_data[k][train_indices]
                    for k in self.feature_data
                }
                y_train = self.labels[train_indices]
                
                X_test = {
                    k: self.feature_data[k][test_indices]
                    for k in self.feature_data
                }
                y_test = self.labels[test_indices]
                
                folds.append((X_train, y_train, X_test, y_test))
        
        return folds

    # -------------------------------------------------------------------------
    # Normalization
    # -------------------------------------------------------------------------

    def normalise_data(self) -> None:
        """
        Normalize feature data (z-score normalization).
        
        Normalizes jet, lepton, and MET features while ignoring padding values.
        Stores normalization factors for later use.
        """
        if self.feature_data is None:
            raise ValueError("Feature data not prepared. Call load_data() first.")
        
        padding_value = self.load_config.padding_value
        
        for data_type in ["lepton", "jet", "met"]:
            if data_type not in self.feature_data:
                continue
            
            data = self.feature_data[data_type]
            
            # Mask for non-padding values
            non_padding_mask = (data != padding_value).all(axis=-1)
            
            # Compute statistics on non-padding data
            mean = np.mean(data[non_padding_mask], axis=0)
            std = np.std(data[non_padding_mask], axis=0)
            
            # Avoid division by zero
            std = np.where(std == 0, 1, std)
            
            # Normalize
            data[non_padding_mask] = (data[non_padding_mask] - mean) / std
            
            # Store normalization factors
            self.data_normalisation_factors[data_type] = {
                "mean": mean,
                "std": std
            }

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------

    def plot_feature_distribution(
        self,
        feature_type: str,
        feature_name: str,
        file_name: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Plot distribution of a feature.
        
        Args:
            feature_type: Type of feature ('jet', 'lepton', etc.)
            feature_name: Name of feature to plot
            file_name: Optional file path to save plot
            **kwargs: Additional arguments passed to plotting function
        """
        data = self.get_feature_data(feature_type, feature_name)
        
        # Flatten and remove padding
        data_flat = data.flatten()
        data_flat = data_flat[data_flat != self.load_config.padding_value]
        
        plt.figure(figsize=(10, 6))
        plt.hist(data_flat, bins=50, **kwargs)
        plt.xlabel(feature_name)
        plt.ylabel("Count")
        plt.title(f"Distribution of {feature_type}.{feature_name}")
        plt.grid(True, alpha=0.3)
        
        if file_name:
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_feature_correlation(self, feature_type: str = "jet") -> None:
        """Plot correlation matrix for features of given type."""
        if self.feature_data is None:
            raise ValueError("Feature data not prepared. Call load_data() first.")
        
        data = self.feature_data[feature_type]
        
        # Flatten spatial dimensions and remove padding
        if data.ndim == 3:
            data = data.reshape(-1, data.shape[-1])
        
        non_padding_mask = (data != self.load_config.padding_value).all(axis=1)
        data = data[non_padding_mask]
        
        # Compute correlation
        corr = np.corrcoef(data.T)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr,
            xticklabels=range(data.shape[1]),
            yticklabels=range(data.shape[1]),
            cmap='coolwarm',
            center=0,
            annot=True,
            fmt='.2f'
        )
        plt.title(f"Feature Correlation Matrix: {feature_type}")
        plt.tight_layout()
        plt.show()