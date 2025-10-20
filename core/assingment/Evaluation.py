import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from typing import Union
from copy import deepcopy

from .Assignment import JetAssignerBase, MLAssignerBase


class MLEvaluatorBase:
    def __init__(self, assigner: MLAssignerBase, X_test, y_test):
        self.assigner = assigner
        self.X_test = X_test
        self.y_test = y_test
        self.max_leptons = assigner.max_leptons
        self.max_jets = assigner.max_jets
        self.global_features = assigner.global_features
        self.n_jets: int = assigner.n_jets
        self.n_leptons: int = assigner.n_leptons
        self.n_global: int = assigner.n_global
        self.padding_value: float = assigner.padding_value
        self.feature_index_dict = assigner.feature_index_dict

    def evaluate(self):
        if self.X_test is None or self.y_test is None:
            raise ValueError(
                "Test data not loaded. Please load test data before evaluation."
            )

        results = self.model.evaluate(self.X_test, self.y_test, verbose=2)
        print("Test Loss and Metrics:", results)
        return results

    def plot_training_history(self):
        if self.assigner.history is None:
            raise ValueError(
                "No training history found. Please train the model before plotting history."
            )

        history = self.assigner.history
        plt.figure(figsize=(12, 4))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="Training Loss")
        if "val_loss" in history.history:
            plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("Loss over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        # Plot accuracy if available
        if "accuracy" in history.history:
            plt.subplot(1, 2, 2)
            plt.plot(history.history["accuracy"], label="Training Accuracy")
            if "val_accuracy" in history.history:
                plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
            plt.title("Accuracy over Epochs")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend()

        plt.tight_layout()
        plt.show()

    def get_assigner_name(self):
        return self.assigner.get_name()

    def compute_permutation_importance(self, num_repeats=5):
        if self.X_test is None or self.y_test is None:
            raise ValueError(
                "Test data not loaded. Please load test data before computing permutation importance."
            )

        baseline_performance = self.assigner.evaluate_accuracy(self.X_test, self.y_test)
        print("Baseline Performance:", baseline_performance)

        importances = {}
        for feature in self.feature_index_dict["jet"]:
            feature_idx = self.feature_index_dict["jet"][feature]
            scores = []
            for _ in range(num_repeats):
                X_permuted = deepcopy(self.X_test)
                X_permuted["jet"][:, :, feature_idx] = np.random.permutation(
                    X_permuted["jet"][:, :, feature_idx]
                )

                permuted_performance = self.assigner.evaluate_accuracy(
                    X_permuted, self.y_test
                )
                scores.append(baseline_performance - permuted_performance)
            importances[feature] = np.mean(scores)

        for feature in self.feature_index_dict["lepton"]:
            feature_idx = self.feature_index_dict["lepton"][feature]
            scores = []
            for _ in range(num_repeats):
                X_permuted = deepcopy(self.X_test)
                X_permuted["lepton"][:, :, feature_idx] = np.random.permutation(
                    X_permuted["lepton"][:, :, feature_idx]
                )
                permuted_performance = self.assigner.evaluate_accuracy(
                    X_permuted, self.y_test
                )
                scores.append(baseline_performance - permuted_performance)
            importances[feature] = np.mean(scores)

        if self.global_features:
            for feature in self.feature_index_dict["global"]:
                feature_idx = self.feature_index_dict["global"][feature]
                scores = []
                for _ in range(num_repeats):
                    X_permuted = deepcopy(self.X_test)
                    X_permuted["global"][:,:, feature_idx] = np.random.permutation(
                        X_permuted["global"][:,:, feature_idx]
                    )
                    permuted_performance = self.assigner.evaluate_accuracy(
                        X_permuted, self.y_test
                    )
                    scores.append(baseline_performance - permuted_performance)
                importances[feature] = np.mean(scores)
        return importances

    def plot_feature_importance(self, importances):
        importances = dict(
            sorted(importances.items(), key=lambda item: item[1], reverse=True)
        )
        features = list(importances.keys())
        scores = list(importances.values())

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(features, scores, color="skyblue")
        ax.set_xlabel("Importance Score")
        ax.set_title("Feature Importance based on Permutation Importance")
        plt.gca().invert_yaxis()
        plt.show()


class JetAssignmentEvaluator:
    def __init__(
        self, assigners: Union[list[JetAssignerBase], JetAssignerBase], X_test, y_test
    ):
        if isinstance(assigners, JetAssignerBase):
            self.assigners = [assigners]
        else:
            self.assigners = assigners

        self.X_test = X_test
        self.y_test = np.argmax(y_test, axis=-2)
        configs = [assigner.config for assigner in self.assigners]
        config_1 = configs[0]
        for config in configs[1:]:
            if config != config_1:
                raise ValueError(
                    "All assigners must have the same DataConfig for consistent evaluation."
                )

        self.config = config_1
        self.feature_index_dict = config_1.get_feature_index_dict()

    def evaluate_all(self):
        results = {}
        for assigner in self.assigners:
            predictions = assigner.predict_indices(self.X_test)
            predicted_indices = np.argmax(predictions, axis=-2)
            accuracy = np.mean(predicted_indices == self.y_test)
            results[assigner.get_name()] = accuracy
            print(f"Accuracy for {assigner.get_name()}: {accuracy}")
        return results

    def plot_all_accuracies(self):
        results = self.evaluate_all()
        names = list(results.keys())
        accuracies = list(results.values())

        plt.figure(figsize=(10, 6))
        plt.bar(names, accuracies)
        plt.xlabel("Assigner")
        plt.ylabel("Accuracy")
        plt.title("Accuracy of Different Jet Assigners")
        plt.ylim(0, 1)
        plt.show()

    @staticmethod
    def _compute_binned_accuracy(binning_mask, accuracy_data, event_weights):
        bin_weights = np.sum(event_weights.reshape(1, -1) * binning_mask, axis=1)
        bin_correct = np.sum(
            accuracy_data.reshape(1, -1) * event_weights.reshape(1, -1) * binning_mask,
            axis=1,
        )
        binned_accuracy = np.divide(
            bin_correct,
            bin_weights,
            out=np.zeros_like(bin_correct, dtype=float),
            where=bin_weights != 0,
        )
        return binned_accuracy

    def plot_binned_accuracy(
        self,
        feature_data_type,
        feature_name,
        bins=20,
        xlims=None,
        fig=None,
        ax=None,
        color_map = None,
        label=None,
    ):
        if feature_data_type not in self.X_test:
            raise ValueError(
                f"Feature data type '{feature_data_type}' not found in test data."
            )
        if feature_name not in self.feature_index_dict[feature_data_type]:
            raise ValueError(
                f"Feature name '{feature_name}' not found in test data for type '{feature_data_type}'."
            )
        if self.X_test[feature_data_type].ndim == 2:
            feature_data = self.X_test[feature_data_type][:, self.feature_index_dict[feature_data_type][feature_name]]
        elif self.X_test[feature_data_type].ndim == 3:
            feature_data = self.X_test[feature_data_type][:, self.feature_index_dict[feature_data_type][feature_name], 0]
        else:
            raise ValueError(
                f"Feature data for type '{feature_data_type}' has unsupported number of dimensions: {self.X_test[feature_data_type].ndim}"
            )
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        if xlims is not None:
            bins = np.linspace(xlims[0], xlims[1], bins + 1)
        else:
            bins = np.linspace(np.min(feature_data), np.max(feature_data), bins + 1)
        binning_mask = (feature_data.reshape(1, -1) >= bins[:-1].reshape(-1, 1)) & (
            feature_data.reshape(1, -1) < bins[1:].reshape(-1, 1)
        )
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        binned_accuracies = {}
        event_weights = self.X_test.get("event_weight", np.ones(feature_data.shape[0]))

        
        for assigner in self.assigners:
            predictions = assigner.predict_indices(self.X_test)
            predicted_indices = np.argmax(predictions, axis=-2)
            accuracy_data = np.all(predicted_indices == self.y_test, axis=-1).astype(
                float
            )
            binned_accuracy = self._compute_binned_accuracy(
                binning_mask, accuracy_data, event_weights
            )
            binned_accuracies[assigner.get_name()] = binned_accuracy

        ax.set_xlabel(feature_name)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.set_xlim(bins[0], bins[-1])
        color_map = plt.get_cmap('tab10') if color_map is None else color_map
        for index, (assigner_name, binned_accuracy) in enumerate(binned_accuracies.items()):
            ax.scatter(
                bin_centers,
                binned_accuracy,
                marker="o",
                label=assigner_name,
                color=color_map(index),
            )
        if label is not None:
            ax.legend(title=label)
        else:
            ax.legend()
        ax.grid()
        ax_clone = ax.twinx()
        bin_counts = np.sum(event_weights.reshape(1, -1) * binning_mask, axis=1)
        ax_clone.bar(
            bin_centers,
            bin_counts,
            width=(bins[1] - bins[0]),
            alpha=0.2,
            color="gray",
            label="Event Count",
        )
        ax_clone.set_ylabel("Event Count")
        plt.title(f"Binned Accuracy vs {feature_name}")
        plt.show()

    def plot_confusion_matrics(self, normalize=True):
        number_assigners = len(self.assigners)
        rows = int(np.ceil(np.sqrt(number_assigners)))
        cols = int(np.ceil(number_assigners / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = axes.flatten()
        for i, assigner in enumerate(self.assigners):
            predictions = assigner.predict_indices(self.X_test)
            predicted_indices = np.argmax(predictions, axis=-2)
            y_true = self.y_test.flatten()
            y_pred = predicted_indices.flatten()
            cm = confusion_matrix(
                y_true, y_pred, normalize="true" if normalize else None
            )
            sns.heatmap(
                cm,
                annot=True,
                fmt=".2f" if normalize else "d",
                ax=axes[i],
                cmap="Blues",
            )
            axes[i].set_title(f"Confusion Matrix: {assigner.get_name()}")
            axes[i].set_xlabel("Predicted Label")
            axes[i].set_ylabel("True Label")
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        return fig, axes[: i + 1]
