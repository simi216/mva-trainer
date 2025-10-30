import tensorflow as tf
import keras
import numpy as np
from abc import ABC, abstractmethod
from core.DataLoader import DataConfig
from core.base_classes import BaseUtilityModel, MLWrapperBase, KerasModelWrapper


class EventReconstructorBase(BaseUtilityModel, ABC):
    def __init__(self, config: DataConfig, name="event_reconstructor", neutrino_reconstruction=False):
        super().__init__(config=config, name=name)
        self.max_jets = config.max_jets
        self.max_leptons = config.max_leptons
        self.neutrino_reconstruction = neutrino_reconstruction
    @abstractmethod
    def predict_indices(self, data_dict):
        pass

    def reconstruct_neutrinos(self, data_dict):
        pass

    def evaluate_accuracy(self, data_dict, true_labels, per_event=False):
        """
        Evaluates the model's performance on the provided data and true indices.

        Args:
            data_dict (dict): A dictionary containing input data for the model.
            true_labels (np.ndarray): The true labels (one-hot) to compare against the model's predictions.
        Returns:
            float | np.ndarray: The accuracy of the model's predictions. If per_event is True,
            returns an array of booleans indicating correctness per event; otherwise, returns overall accuracy.
        """
        predictions = self.predict_indices(data_dict)
        predicted_indices = np.argmax(predictions, axis=-2)
        true_indices = np.argmax(true_labels, axis=-2)
        if per_event:
            correct_predictions = np.all(predicted_indices == true_indices, axis=-1)
        else:
            correct_predictions = np.all(predicted_indices == true_indices, axis=-1)
            accuracy = np.mean(correct_predictions)
        return accuracy

    def evaluate_regression(self, data_dict, true_values):
        """
        Evaluates the regression performance of the model on the provided data and true values.

        Args:
            data_dict (dict): A dictionary containing input data for the model.
            true_values (np.ndarray): The true regression target values to compare against the model's predictions.
        Returns:
            float: The mean squared error of the model's regression predictions.
        """
        regression_predictions = self.reconstruct_neutrinos(data_dict)
        mse = np.mean((regression_predictions - true_values) ** 2)
        return mse

class GroundTruthReconstructor(EventReconstructorBase):
    def __init__(self, config: DataConfig, name="ground_truth_reconstructor"):
        super().__init__(config=config, name=name)
        self.config = config

    def predict_indices(self, data_dict):
        return data_dict["assignment_labels"]

    def reconstruct_neutrinos(self, data_dict):
        if not self.neutrino_reconstruction:
            raise ValueError("Neutrino reconstruction is not enabled for this reconstructor.")
        return data_dict["neutrino_momenta"]

class MLReconstructorBase(EventReconstructorBase, MLWrapperBase):
    def __init__(self, config: DataConfig, name="ml_assigner"):
        super().__init__(config=config, name=name)

    def _build_model_base(self, jet_assignment_probs, regression_output=None, **kwargs):
        jet_assignment_probs.name = "assigment"
        if self.config.has_regression_targets and regression_output is not None:
            regression_output.name = "regression"
            self.model = KerasModelWrapper(
                inputs=[
                    self.inputs["jet_inputs"],
                    self.inputs["lep_inputs"],
                    self.inputs["met_inputs"],
                ],
                outputs={
                    "assigment": jet_assignment_probs,
                    "regression": regression_output,
                },
                **kwargs,
            )
        elif self.config.has_regression_targets and regression_output is None:
            raise ValueError(
                "Regression targets are specified in the config, but regression_output is None."
            )
        elif not self.config.has_regression_targets and regression_output is not None:
            raise Warning(
                "Regression targets are not specified in the config, but regression_output is provided. Ignoring regression_output."
            )
        else:
            self.model = KerasModelWrapper(
                inputs=[
                    self.inputs["jet_inputs"],
                    self.inputs["lep_inputs"],
                    self.inputs["met_inputs"],
                ],
                outputs=jet_assignment_probs,
                **kwargs,
            )

    def compile_model(self, loss, optimizer, metrics=None):
        if self.model is None:
            raise ValueError(
                "Model has not been built yet. Call build_model() before compile_model()."
            )
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def generate_one_hot_encoding(self, predictions, exclusive):
        """
        Generates a one-hot encoded array from the model's predictions.
        This method processes the raw predictions from the model and converts them
        into a one-hot encoded format, indicating the associations between jets and leptons.
        Args:
            predictions (np.ndarray): The raw predictions from the model, typically
                of shape (batch_size, max_jets, max_leptons).
            exclusive (bool): If True, ensures exclusive assignments between jets
                and leptons
        Returns:
            np.ndarray: A one-hot encoded array of shape (batch_size, max_jets, 2),
            where the last dimension represents the association between jets and
            leptons. The value 1 indicates an association, and 0 indicates no association.
        """
        if exclusive:
            one_hot = np.zeros((predictions.shape[0], self.max_jets, 2), dtype=int)
            for i in range(predictions.shape[0]):
                probs = predictions[i].copy()
                for _ in range(self.max_leptons):
                    jet_index, lepton_index = np.unravel_index(
                        np.argmax(probs), probs.shape
                    )
                    one_hot[i, jet_index, lepton_index] = 1
                    probs[jet_index, :] = 0
                    probs[:, lepton_index] = 0
        else:
            one_hot[
                np.arange(predictions.shape[0]),
                np.argmax(predictions[:, :, 0], axis=1),
                0,
            ] = 1
            one_hot[
                np.arange(predictions.shape[0]),
                np.argmax(predictions[:, :, 1], axis=1),
                1,
            ] = 1
        return one_hot

    def predict_indices(self, data: dict[str : np.ndarray], exclusive=True):
        """
        Predicts the indices of jets and leptons based on the model's predictions.
        This method processes the predictions from the model and returns a one-hot
        encoded array indicating the associations between jets and leptons.
        Args:
            data (dict): A dictionary containing input data for prediction. It should
                include keys "jet" and "lepton", and optionally "met" if met
                features are used by the model.
            exclusive (bool, optional): If True, ensures exclusive assignments between
                jets and leptons, where each jet is assigned to at most one lepton and
                vice versa. Defaults to True.
        Returns:
            np.ndarray: A one-hot encoded array of shape (batch_size, max_jets, 2),
            where the last dimension represents the association between jets and
            leptons. The value 1 indicates an association, and 0 indicates no association.
        Raises:
            ValueError: If the model is not built (i.e., `self.model` is None).
        """

        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )
        if self.met_features is not None:
            predictions = self.model.predict_dict(
                [data["jet"], data["lepton"], data["met"]], verbose=0
            )["jet_assignment_probs"]
        else:
            predictions = self.model.predict_dict(
                [data["jet"], data["lepton"]], verbose=0
            )["jet_assignment_probs"]
        one_hot = self.generate_one_hot_encoding(predictions, exclusive)
        return one_hot

    def reconstruct_neutrinos(self, data: dict[str : np.ndarray]):
        """
        Reconstructs neutrino kinematics based on the model's regression output.
        This method processes the regression output from the model and returns
        the reconstructed neutrino kinematics.
        Args:
            data (dict): A dictionary containing input data for prediction. It should
                include keys "jet" and "lepton", and optionally "met" if met
                features are used by the model.
        Returns:
            np.ndarray: An array containing the reconstructed neutrino kinematics.
        Raises:
            ValueError: If the model is not built (i.e., `self.model` is None) or
                if regression targets are not specified in the config.
        """

        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )
        if not self.config.has_regression_targets:
            raise ValueError(
                "Regression targets are not specified in the config."
            )
        if self.met_features is not None:
            regression_predictions = self.model.predict_dict(
                [data["jet"], data["lepton"], data["met"]], verbose=0
            )["regression"]
        else:
            regression_predictions = self.model.predict_dict(
                [data["jet"], data["lepton"]], verbose=0
            )["regression"]
        return regression_predictions