"""
Refactored evaluation module for ML-based reconstruction.

This module provides clean, modular tools for evaluating event reconstruction
methods with support for:
- Feature importance analysis
- Accuracy metrics with bootstrap confidence intervals
- Binned performance analysis
- Complementarity analysis between methods
- Mass resolution calculations
- Comprehensive visualization tools
"""
from .reconstruction_base import EventReconstructorBase, MLReconstructorBase, GroundTruthReconstructor, FixedPrecisionReconstructor
from .ml_evaluator import MLEvaluator, FeatureImportanceCalculator
from .reconstruction_evaluator import ReconstructionEvaluator
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
)
from .physics_calculations import (
    TopReconstructor,
    ResolutionCalculator,
)

__all__ = [
    # Main evaluators
    "MLEvaluator",
    "ReconstructionEvaluator",
    
    # Feature importance
    "FeatureImportanceCalculator",
    
    # Base utilities
    "PlotConfig",
    "BootstrapCalculator",
    "BinningUtility",
    "FeatureExtractor",
    "AccuracyCalculator",
    
    # Plotting utilities
    "AccuracyPlotter",
    "ConfusionMatrixPlotter",
    "ComplementarityPlotter",
    "ResolutionPlotter",
    
    # Physics calculations
    "TopReconstructor",
    "ResolutionCalculator",
]