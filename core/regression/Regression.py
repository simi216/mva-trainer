import tensorflow as tf
import keras
import numpy as np
from abc import ABC, abstractmethod
from core.DataLoader import DataConfig
from core.base_classes import BaseUtilityModel, MLWrapperBase

class NeutrinoRegressionBase(BaseUtilityModel, ABC):
    def __init__(self, config: DataConfig, name="neutrino_regression"):
        super().__init__(config=config, name=name)

    @abstractmethod
    def predict_neutrino_momenta(self, data_dict):
        pass

    def evaluate_regression(self, data_dict, true_momenta):
        """
        Evaluates the model's performance on the provided data and true neutrino momenta.

        Args:
            data_dict (dict): A dictionary containing input data for the model.
            true_momenta (np.ndarray): The true neutrino momenta to compare against the model's predictions.
        Returns:
            float: The mean squared error of the model's predictions.
        """
        predicted_momenta = self.predict_neutrino_momenta(data_dict)
        mse = np.mean((predicted_momenta - true_momenta) ** 2)
        return mse