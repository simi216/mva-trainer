"""
Configuration classes for data loading and ML pipeline.

This module separates loading configuration (LoadConfig) from data structure 
description (DataConfig). LoadConfig is used during data loading, while 
DataConfig describes the loaded data structure and is passed to downstream 
components like ML models.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import numpy as np


@dataclass
class LoadConfig:
    """
    Configuration for loading data from ROOT files.
    
    Contains all options needed during the data loading phase, including
    feature names, truth labels, cuts, and preprocessing options.
    
    This config is used by DataLoader and DataPreprocessor during loading.
    After loading, a DataConfig is created to describe the loaded data structure.
    """
    
    # Feature specifications
    jet_features: List[str]
    lepton_features: List[str]
    jet_truth_label: str
    lepton_truth_label: str
    
    # Maximum objects per event
    max_leptons: int = 2
    max_jets: int = 4
    
    # Optional features
    met_features: Optional[List[str]] = None
    non_training_features: Optional[List[str]] = None
    regression_targets: Optional[List[str]] = None
    event_weight: Optional[str] = None
    
    # Loading options
    padding_value: float = -999.0
    
    def get_feature_clipping_dict(self) -> Dict[str, int]:
        """Build feature clipping configuration for DataLoader."""
        clipping = {
            **{feat: self.max_jets for feat in self.jet_features},
            **{feat: self.max_leptons for feat in self.lepton_features},
            self.jet_truth_label: 6,
            self.lepton_truth_label: 2,
        }
        
        if self.met_features:
            clipping.update({feat: 1 for feat in self.met_features})
        
        if self.non_training_features:
            clipping.update({feat: 1 for feat in self.non_training_features})
        
        if self.event_weight:
            clipping[self.event_weight] = 1
        
        if self.regression_targets:
            clipping.update({target: 1 for target in self.regression_targets})
        
        return clipping
    
    def to_data_config(self) -> 'DataConfig':
        """
        Create a DataConfig from this LoadConfig.
        
        Returns:
            DataConfig describing the structure of loaded data
        """
        return DataConfig(
            jet_features=self.jet_features,
            lepton_features=self.lepton_features,
            met_features=self.met_features,
            non_training_features=self.non_training_features,
            max_jets=self.max_jets,
            max_leptons=self.max_leptons,
            padding_value=self.padding_value,
            has_regression_targets=self.regression_targets is not None,
            regression_target_names=self.regression_targets,
            has_event_weight=self.event_weight is not None,
        )


@dataclass
class DataConfig:
    """
    Configuration describing the structure of loaded data.
    
    This class describes how the data looks after loading, including feature
    names, shapes, and indices. It should be passed to downstream components
    like ML models, training loops, and evaluation scripts.
    
    This config does NOT contain loading-specific options like file paths,
    truth labels, or cuts - those are in LoadConfig.
    
    Attributes:
        jet_features: Names of jet features
        lepton_features: Names of lepton features
        met_features: Names of MET features (optional)
        non_training_features: Names of features not used in training (optional)
        max_jets: Maximum number of jets per event
        max_leptons: Maximum number of leptons per event
        padding_value: Value used for padding
        has_regression_targets: Whether regression targets are present
        regression_target_names: Names of regression targets (optional)
        has_event_weight: Whether event weights are present
        feature_indices: Dictionary mapping feature names to their indices
        data_shapes: Dictionary describing array shapes for each feature type
        custom_features: Custom features added after loading
    """
    
    # Feature names
    jet_features: List[str]
    lepton_features: List[str]
    met_features: Optional[List[str]] = None
    non_training_features: Optional[List[str]] = None
    custom_features: Dict[str, int] = field(default_factory=dict)
    
    # Data structure
    max_jets: int = 4
    max_leptons: int = 2
    padding_value: float = -999.0
    
    # Optional components
    has_regression_targets: bool = False
    regression_target_names: Optional[List[str]] = None
    has_event_weight: bool = False
    
    # Computed properties (populated after loading)
    feature_indices: Dict[str, Dict[str, int]] = field(default_factory=dict, init=False)
    data_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict, init=False)
    custom_features: Dict[str, int] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        """Initialize computed properties."""
        self._build_feature_indices()
        self._build_data_shapes()
    
    def _build_feature_indices(self) -> None:
        """Build index mapping for all feature types."""
        self.feature_indices = {
            "lepton": {var: idx for idx, var in enumerate(self.lepton_features)},
            "jet": {var: idx for idx, var in enumerate(self.jet_features)},
        }
        
        if self.met_features:
            self.feature_indices["met"] = {
                var: idx for idx, var in enumerate(self.met_features)
            }
        
        if self.non_training_features:
            self.feature_indices["non_training"] = {
                var: idx for idx, var in enumerate(self.non_training_features)
            }
        
        if self.has_event_weight:
            self.feature_indices["event_weight"] = {"weight": 0}
        
        if self.has_regression_targets and self.regression_target_names:
            self.feature_indices["regression_targets"] = {
                target: idx for idx, target in enumerate(self.regression_target_names)
            }
    
    def _build_data_shapes(self) -> None:
        """Build expected data shapes for each feature type."""
        # Shape: (n_events, max_leptons, n_features)
        self.data_shapes["lepton"] = (None, self.max_leptons, len(self.lepton_features))
        
        # Shape: (n_events, max_jets, n_features)
        self.data_shapes["jet"] = (None, self.max_jets, len(self.jet_features))
        
        if self.met_features:
            # Shape: (n_events, 1, n_features)
            self.data_shapes["met"] = (None, 1, len(self.met_features))
        
        if self.non_training_features:
            # Shape: (n_events, n_features)
            self.data_shapes["non_training"] = (None, len(self.non_training_features))
        
        if self.has_event_weight:
            # Shape: (n_events,)
            self.data_shapes["event_weight"] = (None,)
        
        if self.has_regression_targets and self.regression_target_names:
            # Shape: (n_events, n_targets)
            self.data_shapes["regression_targets"] = (None, len(self.regression_target_names))
        
        # Shape: (n_events, max_jets, max_leptons)
        self.data_shapes["labels"] = (None, self.max_jets, self.max_leptons)
    
    # =========================================================================
    # Accessors for downstream components (e.g., ML models)
    # =========================================================================
    
    def get_n_jet_features(self) -> int:
        """Get number of jet features."""
        return len(self.jet_features)
    
    def get_n_lepton_features(self) -> int:
        """Get number of lepton features."""
        return len(self.lepton_features)
    
    def get_n_met_features(self) -> int:
        """Get number of MET features."""
        return len(self.met_features) if self.met_features else 0
    
    def get_n_custom_features(self) -> int:
        """Get number of custom features."""
        return len(self.custom_features)
    
    def get_feature_index(self, feature_type: str, feature_name: str) -> int:
        """
        Get the index of a specific feature.
        
        Args:
            feature_type: Type of feature ('jet', 'lepton', 'met', etc.)
            feature_name: Name of the feature
            
        Returns:
            Index of the feature
            
        Raises:
            KeyError: If feature type or name not found
        """
        return self.feature_indices[feature_type][feature_name]
    
    def get_feature_names(self, feature_type: str) -> List[str]:
        """
        Get all feature names of a given type.
        
        Args:
            feature_type: Type of feature ('jet', 'lepton', 'met', etc.)
            
        Returns:
            List of feature names
        """
        if feature_type == "jet":
            return self.jet_features
        elif feature_type == "lepton":
            return self.lepton_features
        elif feature_type == "met":
            return self.met_features if self.met_features else []
        elif feature_type == "non_training":
            return self.non_training_features if self.non_training_features else []
        elif feature_type == "custom":
            return list(self.custom_features.keys())
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
    def get_expected_shape(self, feature_type: str, n_events: Optional[int] = None) -> Tuple[int, ...]:
        """
        Get expected data shape for a feature type.
        
        Args:
            feature_type: Type of feature ('jet', 'lepton', 'met', etc.)
            n_events: Number of events (replaces None in shape)
            
        Returns:
            Expected shape tuple
        """
        shape = self.data_shapes[feature_type]
        if n_events is not None:
            shape = tuple(n_events if s is None else s for s in shape)
        return shape
    
    def validate_data_shape(self, feature_type: str, data: np.ndarray) -> bool:
        """
        Validate that data has the expected shape.
        
        Args:
            feature_type: Type of feature
            data: Data array to validate
            
        Returns:
            True if shape is valid
            
        Raises:
            ValueError: If shape is invalid
        """
        expected_shape = self.get_expected_shape(feature_type, n_events=data.shape[0])
        if data.shape != expected_shape:
            raise ValueError(
                f"Invalid shape for {feature_type}: expected {expected_shape}, got {data.shape}"
            )
        return True
    
    def add_custom_feature(self, name: str, index: int) -> None:
        """
        Register a custom feature.
        
        Args:
            name: Name of the custom feature
            index: Index in the custom feature array
        """
        self.custom_features[name] = index
        if "custom" not in self.feature_indices:
            self.feature_indices["custom"] = {}
        self.feature_indices["custom"][name] = index
    
    def get_model_input_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """
        Get input shapes for ML model (excluding batch dimension).
        
        Returns:
            Dictionary mapping input names to their shapes
        """
        shapes = {
            "jet": (self.max_jets, len(self.jet_features)),
            "lepton": (self.max_leptons, len(self.lepton_features)),
        }
        
        if self.met_features:
            shapes["met"] = (1, len(self.met_features))
        
        if self.custom_features:
            shapes["custom"] = (len(self.custom_features),)
        
        return shapes
    
    def get_output_shape(self) -> Tuple[int, ...]:
        """
        Get output shape for ML model (excluding batch dimension).
        
        Returns:
            Output shape tuple
        """
        if self.has_regression_targets:
            return (len(self.regression_target_names),)
        else:
            # Classification: jet-lepton pairs
            return (self.max_jets, self.max_leptons)
    
    def summary(self) -> str:
        """
        Get a string summary of the data configuration.
        
        Returns:
            Formatted summary string
        """
        lines = [
            "=" * 60,
            "Data Configuration Summary",
            "=" * 60,
            "",
            "Jet Features:",
            f"  Count: {len(self.jet_features)}",
            f"  Max jets: {self.max_jets}",
            f"  Names: {', '.join(self.jet_features)}",
            "",
            "Lepton Features:",
            f"  Count: {len(self.lepton_features)}",
            f"  Max leptons: {self.max_leptons}",
            f"  Names: {', '.join(self.lepton_features)}",
        ]
        
        if self.met_features:
            lines.extend([
                "",
                "MET Features:",
                f"  Count: {len(self.met_features)}",
                f"  Names: {', '.join(self.met_features)}",
            ])
        
        if self.custom_features:
            lines.extend([
                "",
                "Custom Features:",
                f"  Count: {len(self.custom_features)}",
                f"  Names: {', '.join(self.custom_features.keys())}",
            ])
        
        lines.extend([
            "",
            "Data Properties:",
            f"  Padding value: {self.padding_value}",
            f"  Has event weights: {self.has_event_weight}",
            f"  Has regression targets: {self.has_regression_targets}",
        ])
        
        if self.has_regression_targets:
            lines.extend([
                f"  Regression targets: {', '.join(self.regression_target_names)}",
            ])
        
        lines.extend([
            "",
            "Expected Shapes:",
        ])
        for feature_type, shape in self.data_shapes.items():
            lines.append(f"  {feature_type}: {shape}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


# =============================================================================
# Example Usage
# =============================================================================

def create_example_configs():
    """Example showing how to use LoadConfig and DataConfig."""
    
    # 1. Create LoadConfig for data loading
    load_config = LoadConfig(
        jet_features=['jet_pt', 'jet_eta', 'jet_phi', 'jet_m'],
        lepton_features=['lep_pt', 'lep_eta', 'lep_phi'],
        jet_truth_label='jet_truth',
        lepton_truth_label='lep_truth',
        max_jets=4,
        max_leptons=2,
        met_features=['met_pt', 'met_phi'],
        event_weight='event_weight',
        padding_value=-999.0
    )
    
    # 2. Use LoadConfig during data loading
    # preprocessor = DataPreprocessor(load_config)
    # preprocessor.load_data('data.root', 'tree')
    
    # 3. Create DataConfig for downstream use
    data_config = load_config.to_data_config()
    
    # 4. Pass DataConfig to ML model
    # model = MyModel(data_config)
    
    # DataConfig provides useful accessors:
    print(f"Number of jet features: {data_config.get_n_jet_features()}")
    print(f"Jet feature names: {data_config.get_feature_names('jet')}")
    print(f"Expected jet shape: {data_config.get_expected_shape('jet', n_events=1000)}")
    print(f"Model input shapes: {data_config.get_model_input_shapes()}")
    print(f"Model output shape: {data_config.get_output_shape()}")
    
    # Print full summary
    print(data_config.summary())


if __name__ == "__main__":
    create_example_configs()