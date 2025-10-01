import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from core.DataLoader import DataPreprocessor, DataConfig
import core.components as CustomObjects

from .Assignment import JetAssignerBase, MLAssignerBase

class MLEvaluatorBase:
    def __init__(self, assigner: MLAssignerBase):
        self.assigner = assigner
        self.model: keras.Model = assigner.model
        self.X_test = assigner.X_test
        self.y_test = assigner.y_test
        self.max_leptons = assigner.max_leptons
        self.max_jets = assigner.max_jets
        self.global_features = assigner.global_features
        self.n_jets: int = assigner.n_jets
        self.n_leptons: int = assigner.n_leptons
        self.n_global: int = assigner.n_global
        self.padding_value: float = assigner.padding_value
        self.feature_index_dict = assigner.feature_index_dict