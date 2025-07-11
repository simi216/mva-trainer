import tensorflow as tf
import numpy as np
from .DataLoader import DataLoader, DataPreprocessor
from . import CustomObjects

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

import tensorflow as tf
import keras

import sklearn as sk

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow import convert_to_tensor

class BaseModel:
    def __init__(self, data_preprocessor: DataPreprocessor):
        self.model : keras.Model = None
        self.X_train = None
        self.X_test : np.ndarray = None
        self.y_train : np.ndarray = None
        self.y_test : np.ndarray = None
        self.history = None
        self.sample_weights = None
        self.class_weights = None
        self.max_leptons = data_preprocessor.max_leptons
        self.max_jets = data_preprocessor.max_jets
        self.global_features = data_preprocessor.global_features
        self.n_jets: int = len(data_preprocessor.jet_features)
        self.n_leptons: int = len(data_preprocessor.lepton_features)
        self.n_global: int = (
            len(data_preprocessor.global_features)
            if data_preprocessor.global_features
            else 0
        )
        self.padding_value : float = data_preprocessor.padding_value
        self.combined_features = data_preprocessor.combined_features
        self.n_combined: int = data_preprocessor.n_combined
        self.feature_index_dict : dict[str:any] = data_preprocessor.feature_index_dict
        #self.data_preprocessor = data_preprocessor
        self.feature_data : dict[str:np.ndarray] = None

    def load_data(self, X_train = None, y_train = None, X_test = None, y_test = None):
        """
        Load the data into the model.
        """
        if X_train is not None and X_test is not None and y_train is not None and y_test is not None:
            self.X_train = (X_train)
            self.X_test = (X_test)
            self.y_train = (y_train)
            self.y_test = (y_test)
        #else:
        #    self.X_train, self.y_train, self.X_test, self.y_test = self.data_preprocessor.get_data()

    def make_loss_function(self, lambda_excl=0.4):
        @keras.utils.register_keras_serializable(package="CustomUtility")
        def loss(y_true, y_pred):
            """
            y_true: shape (batch_size, 2, num_jets) - one-hot encoded targets
            y_pred: shape (batch_size, 2, num_jets) - softmax outputs
            """
            # --- Cross-Entropy Loss (per lepton) ---
            cross_entropy = keras.losses.categorical_crossentropy(y_true, y_pred, axis = 1)

            # cross_entropy has shape (batch_size, 2); take mean over leptons
            ce_loss = tf.reduce_mean(cross_entropy, axis=-1)

            # --- Soft Exclusion Penalty ---
            # Sum probabilities over leptons (axis=1), shape: (batch_size, num_jets)
            jet_probs_sum = tf.reduce_sum(y_pred, axis=-1)

            # Penalty: ReLU(P_j - 1)^2 for each jet
            violation = tf.nn.relu(jet_probs_sum - 1.0)
            excl_penalty = tf.reduce_sum(tf.square(violation), axis=-1)  # sum over jets

            # --- Total Loss ---
            total_loss = ce_loss + lambda_excl * excl_penalty
            return total_loss
        return loss

    def compile_model(self, lambda_excl = 0.5, *args, **kwargs):
        self.model.compile(
            loss=CustomObjects.AssignmentLoss(lambda_excl=lambda_excl),
            metrics=["accuracy"],
            *args,
            **kwargs,
        )

    def build_model(self):
        raise NotImplementedError("build_model() method must be implemented in subclasses.")

    def summary(self):
        if self.model is None:
            raise ValueError("Model not built. Please build the model using build_model() method.")
        return self.model.summary()


    def train_model(self, epochs=10, batch_size=32, weight=None, *args, **kwargs):
        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )
        if self.X_train is None or self.y_train is None:
            raise ValueError(
                "Data not loaded. Please prepare data using method."
            )
        print("Starting training...")
        if weight is not None:
            if weight == "sample":
                self.compute_sample_weights()
            elif weight == "class":
                self.compute_class_weights()
            else:
                raise ValueError("Invalid weight type. Use 'sample' or 'class'.")

        if self.class_weights is not None:
            kwargs["class_weight"] = (self.class_weights)
        elif self.sample_weights is not None:
            kwargs["sample_weight"] = (self.sample_weights)

        if self.global_features is not None:
            inputs = [self.X_train["jet"], self.X_train["lepton"], self.X_train["global"]]
            validation_data = (
                [self.X_test["jet"], self.X_test["lepton"], self.X_test["global"]],
                (self.y_test)
            )
        else:
            inputs = [self.X_train["jet"], self.X_train["lepton"]]
            validation_data = (
                [self.X_test["jet"], self.X_test["lepton"]],
                (self.y_test)
            )
        self.history = self.model.fit(
            inputs,
            (self.y_train),
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            *args,
            **kwargs,
        )
        
        print("Training completed.")


    def save_model(self, file_path):
        if self.model is None:
            raise ValueError("Model not built. Please build the model using build_model() method.")

        self.model.save(file_path)
        print(f"Model saved to {file_path}")


    def load_model(self, file_path):
        custom_objects={name : obj for name, obj in CustomObjects.__dict__.items() if isinstance(obj, type) and issubclass(obj, keras.layers.Layer)}
        custom_objects["CustomUtility>AssignmentLoss"] = CustomObjects.AssignmentLoss
        custom_objects["CustomUtility>accuracy"] = CustomObjects.accuracy
        self.model = keras.saving.load_model(file_path, custom_objects=custom_objects)
        print(f"Model loaded from {file_path}")


    def predict(self, data):
        if self.model is None:
            raise ValueError("Model not built. Please build the model using build_model() method.")

        predictions = self.model.predict(data, verbose=0)
        return predictions


    def get_model_summary(self):
        if self.model is None:
            raise ValueError("Model not built. Please build the model using build_model() method.")

        return self.model.summary()

    def plot_history(self):
        if self.history is None:
            raise ValueError("No training history available. Please train the model using train_model() method.")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(self.history.history['loss'], label='train_loss')
        ax[0].plot(self.history.history['val_loss'], label='val_loss')
        ax[0].set_title('Loss')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].legend()
        ax[1].plot(self.history.history['accuracy'], label='train_accuracy')
        ax[1].plot(self.history.history['val_accuracy'], label='val_accuracy')
        ax[1].set_title('Accuracy')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()
        return fig, ax

    def get_test_data(self):
        if self.X_test is None or self.y_test is None:
            raise ValueError("Data not split. Please split data using split_data() method.")
        
        return self.X_test, self.y_test


    def compute_sample_weights(self):
        sample_weights = np.ones(self.y_train.shape[0])
        classes = np.unique(self.y_train, axis=0)
        jet_numbers = np.sum(~np.all(self.y_train == self.padding_value, axis=(-1)), axis=-1)

        """for i, class_label in enumerate(classes):
            class_indices = np.where(np.all(self.y_train == class_label, axis=(1,2)))[0]
            class_weight = 1.0 / len(class_indices) * self.y_train.shape[0]
            sample_weights[class_indices] = np.sqrt(class_weight)"""
        
        for i, jet_number in enumerate(np.unique(jet_numbers)):
            jet_indices = np.where(jet_numbers == jet_number)[0]
            if len(jet_indices) > 0:
                class_weight = 1.0 / len(jet_indices) * self.y_train.shape[0]
                sample_weights[jet_indices] = np.sqrt(class_weight)

        sample_weights /= np.mean(sample_weights)

        if self.X_train["event_weight"] is not None:
            sample_weights *= self.X_train["event_weight"] / np.mean(self.X_train["event_weight"])

        if self.sample_weights is not None:
            self.sample_weights *= sample_weights
        else:
            self.sample_weights = sample_weights


    def enhance_region(self, variable, data_type, low_cut = None, high_cut = None,  factor = 1.0):
        if self.sample_weights is None:
            self.sample_weights = np.ones(self.y_train.shape[0])
        if data_type not in self.feature_index_dict:
            raise ValueError(
                f"Data type {data_type} not found. Use 'jet', 'lepton', 'non_training', or 'global'."
            )
        if variable not in self.feature_index_dict[data_type]:
            raise ValueError(
                f"Feature {variable} not found in {data_type} features."
            )
        if low_cut is None and high_cut is None:
            raise ValueError(
                "At least one of low_cut or high_cut must be provided."
            )
        if low_cut is not None and high_cut is not None:
            self.sample_weights[
                self.X_train[data_type][:, self.feature_index_dict[data_type][variable]] < high_cut & self.X_train[data_type][:, self.feature_index_dict[data_type][variable]] > low_cut
            ] *= factor
        elif low_cut is not None:
            self.sample_weights[
                self.X_train[data_type][:, self.feature_index_dict[data_type][variable]] > low_cut
            ] *= factor
        elif high_cut is not None:
            self.sample_weights[
                self.X_train[data_type][:, self.feature_index_dict[data_type][variable]] < high_cut
            ] *= factor

    def save_model(self, file_path="model.keras"):
        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )
        if ".keras" not in file_path:
            raise ValueError(
                "File path must end with .keras. Please provide a valid file path."
            )
        self.model.save(file_path)
        def myprint(s):
            with open(file_path.replace(".keras", "_structure.txt"),'a') as f:
                print(s, file=f)
        self.model.summary(print_fn=myprint)

        print(f"Model saved to {file_path}.")


    def predict_indices(self, data, exclusive = True):
        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )
        if self.global_features is not None:
            predictions = self.model.predict(
                [data["jet"], data["lepton"], data["global"]], verbose=0
            )
        else:
            predictions = self.model.predict([data["jet"], data["lepton"]],verbose=0)
        batch_size = predictions.shape[0]
        one_hot = np.zeros((batch_size, self.max_jets, 2), dtype=int)
        if exclusive:
            for i in range(batch_size):
                probs = predictions[i].copy()
                for _ in range(self.max_leptons):
                    jet_index, lepton_index = np.unravel_index(np.argmax(probs), probs.shape)
                    one_hot[i, jet_index, lepton_index] = 1
                    probs[jet_index, : ] = 0
                    probs[:, lepton_index] = 0
        else:
            one_hot[np.arange(batch_size), np.argmax(predictions[:, :, 0], axis=1), 0] = 1
            one_hot[np.arange(batch_size), np.argmax(predictions[:, :, 1], axis=1), 1] = 1
        return one_hot


    def duplicate_jets(self):
        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )
        if self.X_test is None or self.y_test is None:
            raise ValueError(
                "Data not loaded. Please prepare data using method."
            )

        if self.global_features is not None:
            y_pred = self.model.predict(
                [self.X_test["jet"], self.X_test["lepton"], self.X_test["global"]], verbose=0
            )
        else:
            y_pred = self.model.predict([self.X_test["jet"], self.X_test["lepton"]], verbose=0)

        y_pred = self.predict_indices(self.X_test, exclusive=False)
        return np.mean(np.argmax(y_pred[:, :, 0],axis = 1) == np.argmax(y_pred[:, :, 1], axis = 1), axis=0)


    def plot_history(self):
        if self.history is None:
            print("No training history available.")
            return 
        keys = self.history.history.keys()
        accept_keys = []
        for key in keys:
            if "val" not in key and "val_" + key in keys:
                accept_keys.append(key)
        fig, ax = plt.subplots(len(accept_keys), 1, figsize=(5, 5 * len(accept_keys)))
        for i, key in enumerate(accept_keys):
            ax[i].plot(self.history.history[key], label="train")
            ax[i].plot(self.history.history["val_" + key], label="validation")
            ax[i].set_title(key)
            ax[i].set_xlabel("Epochs")
            ax[i].set_ylabel(key)
            ax[i].legend()
        return fig, ax

    def compute_permutation_importance(self, shuffle_number = 1):
        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )
        if self.y_test is None:
            raise ValueError(
                "Labels not prepared. Please prepare data using prepare_data() method."
            )
        if self.X_test is None:
            raise ValueError(
                "Feature data not prepared. Please prepare data using prepare_data() method."
            )
        features = list(self.feature_index_dict["jet"].keys()) + list(
            self.feature_index_dict["lepton"].keys()
        )
        if self.global_features is not None:
            features += list(self.feature_index_dict["global"].keys())

        importance_scores_lep_1 = pd.DataFrame(index=features, columns=["mean", "std"])
        importance_scores_lep_2 = pd.DataFrame(index=features, columns=["mean", "std"])
        importance_scores_combined = pd.DataFrame(
            index=features, columns=["mean", "std"]
        )
        ground_truth = self.y_test.copy()
        lep_1_truth = np.argmax(ground_truth[:, :, 0], axis=1)
        lep_2_truth = np.argmax(ground_truth[:, :, 1], axis=1)
        lep_1_pred = np.argmax(self.y_test[:, :, 0], axis=1)
        lep_2_pred = np.argmax(self.y_test[:, :, 1], axis=1)
        lep_1_base_accuracy = np.mean(lep_1_truth == lep_1_pred)
        lep_2_base_accuracy = np.mean(lep_2_truth == lep_2_pred)
        combined_base_accuracy = np.mean(
            (lep_1_truth == lep_1_pred) & (lep_2_truth == lep_2_pred)
        )
        lep_1_accuracy = np.zeros(shuffle_number)
        lep_2_accuracy = np.zeros(shuffle_number)
        combined_accuracy = np.zeros(shuffle_number)

        for shuffle_index in range(shuffle_number):

            for feature in features:
                X_test_permuted = {}
                for data in self.X_test:
                    X_test_permuted[data] = self.X_test[data].copy()
                if feature in self.feature_index_dict["jet"]:
                    feature_index = self.feature_index_dict["jet"][feature]
                    mask = X_test_permuted["jet"][:, :, feature_index] != self.padding_value
                    permuted_values = np.random.permutation(
                        X_test_permuted["jet"][:, :, feature_index][mask]
                    )
                    X_test_permuted["jet"][:, :, feature_index][mask] = permuted_values
                elif feature in self.feature_index_dict["lepton"]:
                    feature_index = self.feature_index_dict["lepton"][feature]
                    X_test_permuted["lepton"][:,:, feature_index] = np.random.permutation(
                        X_test_permuted["lepton"][:,:, feature_index]
                    )
                elif feature in self.feature_index_dict["global"]:
                    feature_index = self.feature_index_dict["global"][feature]
                    X_test_permuted["global"][:, :, feature_index] = np.random.permutation(
                        X_test_permuted["global"][:,:, feature_index]
                    )
                else:
                    raise ValueError(
                        f"Feature {feature} not found in jet, lepton, or global features."
                    )

                if self.global_features is not None:
                    y_pred_permuted = self.model.predict(
                        [
                            X_test_permuted["jet"],
                            X_test_permuted["lepton"],
                            X_test_permuted["global"],
                        ],
                        verbose=0,
                    )
                else:
                    y_pred_permuted = self.model.predict(
                        [X_test_permuted["jet"], X_test_permuted["lepton"]], verbose=0
                    )
                y_pred_permuted = self.predict_indices(X_test_permuted)
                lep_1_pred = np.argmax(y_pred_permuted[:,:,0], axis=1)
                lep_2_pred = np.argmax(y_pred_permuted[:,:,1], axis=1)
                lep_1_accuracy[shuffle_index] = np.mean(lep_1_pred == lep_1_truth)
                lep_2_accuracy[shuffle_index] = np.mean(lep_2_pred == lep_2_truth)
                combined_accuracy[shuffle_index] = np.mean(
                    (lep_1_pred == lep_1_truth) & (lep_2_pred == lep_2_truth)
                )
            print(f"Feature: {feature} computation done.")
            importance_scores_lep_1.loc[feature, "mean"] = np.mean(
                lep_1_base_accuracy - lep_1_accuracy
            )
            importance_scores_lep_1.loc[feature, "std"] = np.std(
                lep_1_base_accuracy - (lep_1_accuracy)
            ) / np.sqrt(shuffle_number)
            importance_scores_lep_2.loc[feature, "mean"] = (
                lep_2_base_accuracy - np.mean(lep_2_accuracy)
            )
            importance_scores_lep_2.loc[feature, "std"] = np.std(
                lep_2_base_accuracy - (lep_2_accuracy)
            ) / np.sqrt(shuffle_number)
            importance_scores_combined.loc[feature, "mean"] = (
                combined_base_accuracy - np.mean(combined_accuracy)
            )
            importance_scores_combined.loc[feature, "std"] = np.std(
                combined_base_accuracy - (combined_accuracy)
            ) / np.sqrt(shuffle_number)
        return importance_scores_lep_1, importance_scores_lep_2, importance_scores_combined


    def plot_permutation_importance(self, shuffle_number, file_name=None):
        importance_scores_lep_1, importance_scores_lep_2, importance_scores_combined = self.compute_permutation_importance(shuffle_number)
        importance_scores_lep_1 = importance_scores_lep_1.sort_values(ascending=False, by="mean")
        importance_scores_lep_2 = importance_scores_lep_2.sort_values(ascending=False, by="mean")
        importance_scores_combined = importance_scores_combined.sort_values(ascending=False, by="mean")
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        importance_scores_lep_1["mean"].plot(
            kind="bar",
            yerr=importance_scores_lep_1["std"],
            ax=ax[0],
            color="blue",
            alpha=0.7,
        )
        importance_scores_lep_2["mean"].plot(
            kind="bar",
            yerr=importance_scores_lep_2["std"],
            ax=ax[1],
            color="orange",
            alpha=0.7,
        )
        importance_scores_combined["mean"].plot(
            kind="bar",
            yerr=importance_scores_combined["std"],
            ax=ax[2],
            color="green",
            alpha=0.7,
        )
        ax[0].set_title("Lepton 1 Accuracy")
        ax[1].set_title("Lepton 2 Accuracy")
        ax[2].set_title("Combined Accuracy")
        ax[0].set_ylabel("Importance Score")
        ax[1].set_ylabel("Importance Score")
        ax[2].set_ylabel("Importance Score")
        fig.tight_layout()
        if file_name is not None:
            fig.savefig(file_name)
            print(f"Permutation importance plotted and saved to {file_name}.")
        return fig, ax


    def accuracy_vs_feature(self, feature_name, data_type = "non_training", bins=50, xlims=None):
        """
        Plot accuracy vs a feature, given truth and prediction arrays.

        Args:
            feature_name: str, name of the feature (for axis labeling)
            xlims: tuple (min, max) for feature range (optional)
            bins: int, number of bins

        Returns:
            fig, ax: matplotlib figure and axes
        """
        if data_type not in self.feature_index_dict:
            raise ValueError(
                f"Data type {data_type} not found. Use 'jet', 'lepton', 'non_training', or 'global'."
            )

        if feature_name not in self.feature_index_dict[data_type]:
            raise ValueError(
                f"Feature {feature_name} not found in non-training features."
            )

        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )

        if self.global_features is not None:
            y_pred = self.model.predict(
                [
                    self.X_test["jet"],
                    self.X_test["lepton"],
                    self.X_test["global"],
                ], verbose=0
            )
        else:
            y_pred = self.model.predict(
                [self.X_test["jet"], self.X_test["lepton"]],
                verbose=0
            )
        y_pred = self.predict_indices(self.X_test)
        feature_data = self.X_test[data_type][:, self.feature_index_dict[data_type][feature_name]]
        prediction = np.array([np.argmax(y_pred[:,:,0], axis=1), np.argmax(y_pred[:,:,1],axis = 1)], dtype=int).T

        return self.plot_accuracy_feature(
            np.argmax(self.y_test, axis=1),
            prediction,
            feature_data,
            feature_name,
            bins=bins,
            xlims=xlims,
            label = "RNN Matcher",
            event_weights=self.X_test["event_weight"],
            )


    def evaluate_accuracy(self):
        return self.compute_accuracy(np.argmax(self.y_test, axis = 1), np.argmax(self.predict_indices(self.X_test), axis = 1))


    @staticmethod
    def compute_accuracy(truth, prediction):
        lep_1_truth, lep_2_truth = truth[:, 0], truth[:, 1]
        lep_1_pred, lep_2_pred = prediction[:, 0], prediction[:, 1]

        lep_1_accuracy = np.mean(lep_1_truth == lep_1_pred)
        lep_2_accuracy = np.mean(lep_2_truth == lep_2_pred)
        combined_accuracy = np.mean((lep_1_truth == lep_1_pred) & (lep_2_truth == lep_2_pred))

        return lep_1_accuracy, lep_2_accuracy, combined_accuracy

    def get_binned_accuracy(self, feature_name, data_type = "non_training", bins=20, xlims=None):
        """
        Compute binned accuracy for a feature.

        Args:
            feature_name: str, name of the feature (for axis labeling)
            xlims: tuple (min, max) for feature range (optional)
            bins: int, number of bins

        Returns:
            binned_accuracy_lep_1, binned_accuracy_lep_2, binned_accuracy_combined: numpy arrays
        """
        if data_type not in self.feature_index_dict:
            raise ValueError(
                f"Data type {data_type} not found. Use 'jet', 'lepton', 'non_training', or 'global'."
            )

        if feature_name not in self.feature_index_dict[data_type]:
            raise ValueError(
                f"Feature {feature_name} not found in non-training features."
            )

        if self.model is None:
            raise ValueError(
                "Model not built. Please build the model using build_model() method."
            )

        if self.global_features is not None:
            y_pred = self.model.predict(
                [
                    self.X_test["jet"],
                    self.X_test["lepton"],
                    self.X_test["global"],
                ],
                verbose=0
            )
        else:
            y_pred = self.model.predict(
                [self.X_test["jet"], self.X_test["lepton"]],
                verbose=0
            )
        y_pred = self.predict_indices(self.X_test)
        feature_data = self.X_test[data_type][:, self.feature_index_dict[data_type][feature_name]]
        prediction = np.array([np.argmax(y_pred[:,:,0], axis=1), np.argmax(y_pred[:,:,1],axis = 1)], dtype=int).T

        return BaseModel.compute_binned_accuracy(
            np.argmax(self.y_test, axis=1),
            prediction,
            feature_data,
            bins=bins,
            xlims=xlims,
            event_weights=self.X_test["event_weight"],
        )

    @staticmethod
    def compute_binned_accuracy(truth : np.ndarray, prediction : np.ndarray, feature_data : np.ndarray, bins=20, xlims=None, event_weights = None):
        lep_1_truth, lep_2_truth = truth[:, 0], truth[:, 1]
        lep_1_pred, lep_2_pred = prediction[:, 0], prediction[:, 1]
        if xlims is None:
            feature_hist, feature_bins = np.histogram(
                feature_data, bins=bins, density=True, weights=event_weights
            )
        else:
            feature_hist, feature_bins = np.histogram(
                feature_data, bins=bins, density=True, weights=event_weights, range=xlims,
            )

        xlims = (feature_bins[0], feature_bins[-1])

        binning_mask = ((feature_data.reshape(-1, 1) <= feature_bins[1:].reshape(1, -1)) & \
                       (feature_data.reshape(-1, 1) >= feature_bins[:-1].reshape(1, -1))).astype(float)
        binned_accuracy_combined = np.sum(
            np.broadcast_to(((lep_1_pred == lep_1_truth) & (lep_2_pred == lep_2_truth)).reshape(-1, 1), (lep_2_pred.shape[0], bins)) * binning_mask, axis=0
        ) / (np.sum(binning_mask, axis=0) + 1e-9)

        return  binned_accuracy_combined, feature_bins, feature_hist

    @staticmethod
    def plot_accuracy_feature(truth, prediction, feature_data, feature_name, bins=20, xlims = None, fig = None, ax = None, accuracy_color = "tab:blue",label = None, event_weights = None):
        import matplotlib.pyplot as plt

        # Use compute_binned_accuracy to get binned accuracies
        binned_accuracy_combined, feature_bins, feature_hist = BaseModel.compute_binned_accuracy(truth, prediction, feature_data, bins=bins, xlims=xlims, event_weights = None)
        # Compute histogram for feature counts (optionally weighted)
        centers = (feature_bins[:-1] + feature_bins[1:]) / 2
        # Bootstrapping for error bars
        n_bootstrap = 10
        rng = np.random.default_rng()
        acc_comb_boot = np.zeros((n_bootstrap, bins))
        for i in range(n_bootstrap):
            indices = rng.integers(0, len(feature_data), len(feature_data))
            truth_bs = truth[indices]
            pred_bs = prediction[indices]
            feat_bs = feature_data[indices]
            acc_comb_boot[i], _ , _ = BaseModel.compute_binned_accuracy(
            truth_bs, pred_bs, feat_bs, bins=bins, xlims=(feature_bins[0], feature_bins[-1])
            )
        err_comb = np.std(acc_comb_boot, axis=0) / np.sqrt(n_bootstrap)

        # Prepare data for plotting

        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            # Plot histogram of feature counts on secondary y-axis
            ax_clone = ax.twinx()
            ax_clone.set_ylabel("Feature Count", color='tab:orange')
            ax_clone.bar(
            centers,
            feature_hist,
            width=np.diff(feature_bins),
            alpha=0.3,
            color='tab:orange'
            )
            ax_clone.tick_params(axis="y", labelcolor='tab:orange')
            ax.set_xlabel(feature_name)
            ax.set_ylim(0, 1.1)
            ax.set_ylabel("Accuracy")
            ax.tick_params(axis="y")
            ax.set_xlim(xlims)
        # Plot accuracy with error bars
        ax.errorbar(
        centers,
        binned_accuracy_combined,
        yerr=err_comb,
        label=label,
        fmt='o',
        capsize=2,
        color=accuracy_color
        )

        fig.tight_layout()
        return fig, ax

    @staticmethod
    def plot_external_confusion_matrix(jet_matcher_indices = None, truth_index = None, max_jets = None, n_bootstrap=100, sample_weight = None):
        bootstrap_confusion_lep_1 = np.zeros((n_bootstrap, max_jets, max_jets))
        bootstrap_confusion_lep_2 = np.zeros((n_bootstrap, max_jets, max_jets))

        for i in range(n_bootstrap):
            indices = np.random.choice(len(truth_index), len(truth_index), replace=True)
            truth_resampled = truth_index[indices]
            matcher_resampled = jet_matcher_indices[indices]
            if sample_weight is not None:
                sample_weight_resampled = sample_weight[indices]
            else:
                sample_weight_resampled = None

            confusion_lep_1_resampled = confusion_matrix(truth_resampled[:, 0], matcher_resampled[:, 0], labels=np.arange(max_jets), sample_weight=sample_weight_resampled)
            confusion_lep_2_resampled = confusion_matrix(truth_resampled[:, 1], matcher_resampled[:, 1], labels=np.arange(max_jets), sample_weight=sample_weight_resampled)

            bootstrap_confusion_lep_1[i] = confusion_lep_1_resampled.astype('float') / confusion_lep_1_resampled.sum(axis=1)[:, np.newaxis]
            bootstrap_confusion_lep_2[i] = confusion_lep_2_resampled.astype('float') / confusion_lep_2_resampled.sum(axis=1)[:, np.newaxis]

        mean_confusion_lep_1 = bootstrap_confusion_lep_1.mean(axis=0)
        mean_confusion_lep_2 = bootstrap_confusion_lep_2.mean(axis=0)
        std_confusion_lep_1 = bootstrap_confusion_lep_1.std(axis=0)
        std_confusion_lep_2 = bootstrap_confusion_lep_2.std(axis=0)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        baseline_confusion_lep_1 = confusion_matrix(truth_index[:, 0], jet_matcher_indices[:, 0], labels=np.arange(max_jets), sample_weight=sample_weight_resampled).astype('float') / confusion_matrix(truth_index[:, 0], jet_matcher_indices[:, 0], labels=np.arange(max_jets), sample_weight=sample_weight_resampled).sum(axis=1)[:, np.newaxis]
        baseline_confusion_lep_2 = confusion_matrix(truth_index[:, 1], jet_matcher_indices[:, 1], labels=np.arange(max_jets), sample_weight=sample_weight_resampled).astype('float') / confusion_matrix(truth_index[:, 1], jet_matcher_indices[:, 1], labels=np.arange(max_jets), sample_weight=sample_weight_resampled).sum(axis=1)[:, np.newaxis]
        sns.heatmap(baseline_confusion_lep_1, annot=True, fmt='.2f', cmap='Blues', cbar=False, ax=ax[0])
        sns.heatmap(baseline_confusion_lep_2, annot=True, fmt='.2f', cmap='Blues', cbar=False, ax=ax[1])

        for i in range(max_jets):
            for j in range(max_jets):
                ax[0].text(j + 0.5 + 0.3, i + 0.5, f"±{std_confusion_lep_1[i, j]:.2f}", color="red", ha="center", va="center", fontsize=8)
                ax[1].text(j + 0.5 + 0.3, i + 0.5, f"±{std_confusion_lep_2[i, j]:.2f}", color="red", ha="center", va="center", fontsize=8)

        ax[0].set_title('Confusion Matrix for Lepton 1 (Bootstrap)')
        ax[1].set_title('Confusion Matrix for Lepton 2 (Bootstrap)')
        ax[0].set_xlabel('Predicted Label')
        ax[1].set_xlabel('Predicted Label')
        ax[0].set_ylabel('True Label')
        ax[1].set_ylabel('True Label')
        return fig ,ax


    def plot_confusion_matrix(self, n_bootstrap=100, exclusive = True):
        jet_matcher_indices = np.argmax(self.predict_indices(self.X_test, exclusive=exclusive), axis = 1)
        truth_index = np.argmax(self.y_test, axis=1)
        max_jets = self.max_jets
        return self.plot_external_confusion_matrix(jet_matcher_indices, truth_index, max_jets, n_bootstrap=n_bootstrap, sample_weight=self.X_test["event_weight"])


    def get_confusion_matrix(self, data = None, labels = None, sample_weights = None, exclusive = True):
        if data is None and labels is None and sample_weights is None:
            data = self.X_test
            labels = self.y_test
            if "event_weight" in self.X_test:
                sample_weights = self.X_test["event_weight"]
        elif data is None or labels is None:
            raise ValueError("Both data and labels must be provided or both must be None.")

        jet_matcher_indices = np.argmax(self.predict_indices(data, exclusive=exclusive), axis=1)            
        truth_index = np.argmax(labels, axis=1)
        max_jets = self.max_jets
        lep_1_confusion_matrix = confusion_matrix(truth_index[:, 0], jet_matcher_indices[:, 0], labels=np.arange(max_jets), sample_weight=sample_weights).astype('float')
        lep_2_confusion_matrix = confusion_matrix(truth_index[:, 1], jet_matcher_indices[:, 1], labels=np.arange(max_jets), sample_weight=sample_weights).astype('float')
        normed_lep_1_confusion_matrix = lep_1_confusion_matrix / lep_1_confusion_matrix.sum(axis=1)[:, np.newaxis]
        normed_lep_2_confusion_matrix = lep_2_confusion_matrix / lep_2_confusion_matrix.sum(axis=1)[:, np.newaxis]

        return normed_lep_1_confusion_matrix, normed_lep_2_confusion_matrix
