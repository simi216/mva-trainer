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
    NUM_LEPTONS: int = 2
    max_jets: int = 4

    # Optional features
    met_features: Optional[List[str]] = None
    non_training_features: Optional[List[str]] = None
    neutrino_momentum_features: Optional[List[str]] = None
    antineutrino_momentum_features: Optional[List[str]] = None
    nu_flows_neutrino_momentum_features: Optional[List[str]] = None
    nu_flows_antineutrino_momentum_features: Optional[List[str]] = None
    event_weight: Optional[str] = None
    mc_event_number: Optional[str] = None

    # Loading options
    padding_value: float = -999.0
    def get_feature_clipping_dict(self) -> Dict[str, int]:
        """Build feature clipping configuration for DataLoader."""
        clipping = {
            **{feat: self.max_jets for feat in self.jet_features},
            **{feat: self.NUM_LEPTONS for feat in self.lepton_features},
            self.jet_truth_label: 6,
            self.lepton_truth_label: 2,
        }
        clipping.update({feat: 1 for feat in self.met_features})

        if self.non_training_features:
            clipping.update({feat: 1 for feat in self.non_training_features})

        if self.event_weight:
            clipping[self.event_weight] = 1
        if self.mc_event_number:
            clipping[self.mc_event_number] = 1

        if (not self.neutrino_momentum_features is None) ^ (not self.antineutrino_momentum_features is None):
            raise ValueError(
                "Both neutrino_momentum_features and "
                "antineutrino_momentum_features must be provided together."
            )
        elif self.neutrino_momentum_features and self.antineutrino_momentum_features:
            if len(self.neutrino_momentum_features) != len(
                self.antineutrino_momentum_features
            ):
                raise ValueError(
                    "neutrino_momentum_features and "
                    "antineutrino_momentum_features must have the same length."
                )
            clipping.update({feat: 1 for feat in self.neutrino_momentum_features})
            clipping.update({feat: 1 for feat in self.antineutrino_momentum_features})
        
        if (not self.nu_flows_neutrino_momentum_features is None) ^ (not self.nu_flows_antineutrino_momentum_features is None):
            raise ValueError(
                "Both nu_flows_neutrino_momentum_features and "
                "nu_flows_antineutrino_momentum_features must be provided together."
            )
        elif self.nu_flows_neutrino_momentum_features and self.nu_flows_antineutrino_momentum_features:
            if len(self.nu_flows_neutrino_momentum_features) != len(
                self.nu_flows_antineutrino_momentum_features
            ):
                raise ValueError(
                    "nu_flows_neutrino_momentum_features and "
                    "nu_flows_antineutrino_momentum_features must have the same length."
                )
            clipping.update({feat: 1 for feat in self.nu_flows_neutrino_momentum_features})
            clipping.update({feat: 1 for feat in self.nu_flows_antineutrino_momentum_features})


        return clipping

    def to_data_config(self) -> "DataConfig":
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
            NUM_LEPTONS=self.NUM_LEPTONS,
            padding_value=self.padding_value,
            has_regression_targets=self.neutrino_momentum_features is not None,
            neutrino_momentum_features=self.neutrino_momentum_features,
            antineutrino_momentum_features=self.antineutrino_momentum_features,
            has_nu_flows_regression_targets=self.nu_flows_neutrino_momentum_features is not None,
            nu_flows_neutrino_momentum_features=self.nu_flows_neutrino_momentum_features,
            nu_flows_antineutrino_momentum_features=self.nu_flows_antineutrino_momentum_features,
            has_event_weight=self.event_weight is not None,
            has_event_number=self.mc_event_number is not None,
        )
    
def get_load_config_from_yaml(yaml_path: str) -> LoadConfig:
    """
    Load a LoadConfig from a YAML file.

    Args:
        yaml_path: Path to the YAML configuration file
    Returns:
        LoadConfig instance
    """
    import yaml

    with open(yaml_path, "r") as f:
        yaml_dict = yaml.safe_load(f)
    config_dict = yaml_dict.get("LoadConfig", {})

    return LoadConfig(**config_dict)


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
        NUM_LEPTONS: Maximum number of leptons per event
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

    # Non-training features
    non_training_features: Optional[List[str]] = None
    custom_features: Dict[str, int] = field(default_factory=dict)

    # Data structure
    max_jets: int = 4
    NUM_LEPTONS: int = 2
    padding_value: float = -999.0

    # Optional components
    has_regression_targets: bool = False
    neutrino_momentum_features: Optional[List[str]] = None
    antineutrino_momentum_features: Optional[List[str]] = None

    # Nu-flows regression targets
    has_nu_flows_regression_targets: bool = False
    nu_flows_neutrino_momentum_features: Optional[List[str]] = None
    nu_flows_antineutrino_momentum_features: Optional[List[str]] = None

    

    has_event_weight: bool = False
    has_event_number: bool = False

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

        self.feature_indices["met"] = {
            var: idx for idx, var in enumerate(self.met_features)
        }

        if self.non_training_features:
            self.feature_indices["non_training"] = {
                var: idx for idx, var in enumerate(self.non_training_features)
            }

        if self.has_event_weight:
            self.feature_indices["event_weight"] = {"weight": 0}
        
        if self.has_event_number:
            self.feature_indices["event_number"] = {"event_number": 0}

        if self.has_regression_targets and self.neutrino_momentum_features:
            self.feature_indices["regression_targets"] = {
                target: idx
                for idx, target in enumerate(self.neutrino_momentum_features)
            }
        
        if self.has_regression_targets and self.nu_flows_neutrino_momentum_features:
            self.feature_indices["nu_flows_regression_targets"] = {
                target: idx
                for idx, target in enumerate(self.nu_flows_neutrino_momentum_features)
            }

        

    def _build_data_shapes(self) -> None:
        """Build expected data shapes for each feature type."""
        # Shape: (n_events, NUM_LEPTONS, n_features)
        self.data_shapes["lepton"] = (None, self.NUM_LEPTONS, len(self.lepton_features))

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
        
        if self.has_event_number:
            # Shape: (n_events,)
            self.data_shapes["event_number"] = (None,)

        if self.has_regression_targets:
            # Shape: (n_events, n_targets)
            self.data_shapes["regression_targets"] = (
                None,
                self.NUM_LEPTONS,
                len(self.neutrino_momentum_features),
            )

        # Shape: (n_events, max_jets, NUM_LEPTONS)
        self.data_shapes["labels"] = (None, self.max_jets, self.NUM_LEPTONS)

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
    
    def get_n_regression_targets(self) -> int:
        """Get number of regression target features."""
        return len(self.neutrino_momentum_features) if self.neutrino_momentum_features else 0

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

    def get_expected_shape(
        self, feature_type: str, n_events: Optional[int] = None
    ) -> Tuple[int, ...]:
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
            "lepton": (self.NUM_LEPTONS, len(self.lepton_features)),
        }

        if self.met_features:
            shapes["met"] = (1, len(self.met_features))

        return shapes

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
            f"  Max leptons: {self.NUM_LEPTONS}",
            f"  Names: {', '.join(self.lepton_features)}",
        ]

        if self.met_features:
            lines.extend(
                [
                    "",
                    "MET Features:",
                    f"  Count: {len(self.met_features)}",
                    f"  Names: {', '.join(self.met_features)}",
                ]
            )

        if self.custom_features:
            lines.extend(
                [
                    "",
                    "Custom Features:",
                    f"  Count: {len(self.custom_features)}",
                    f"  Names: {', '.join(self.custom_features.keys())}",
                ]
            )

        lines.extend(
            [
                "",
                "Data Properties:",
                f"  Padding value: {self.padding_value}",
                f"  Has event weights: {self.has_event_weight}",
                f"  Has regression targets: {self.has_regression_targets}",
            ]
        )

        if self.has_regression_targets:
            lines.extend(
                [
                "Neutrino Momentum Features:",
                f"  Count: {len(self.neutrino_momentum_features)}",
                f" Max neutrinos: {self.NUM_LEPTONS}",
                f"  Names: {', '.join(self.neutrino_momentum_features)}",
                ]
            )

        lines.extend(
            [
                "",
                "Expected Shapes:",
            ]
        )
        for feature_type, shape in self.data_shapes.items():
            lines.append(f"  {feature_type}: {shape}")

        lines.append("=" * 60)

        return "\n".join(lines)
