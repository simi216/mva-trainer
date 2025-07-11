from .BaseModel import BaseModel
from .DataLoader import DataLoader, DataPreprocessor

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import threading
import copy


class KFoldEvaluation:
    def __init__(self, model_type : type[BaseModel], data_processor : DataPreprocessor , n_folds = 5, n_splits=1, random_state = 42):
        self.model_type = model_type
        self.feature_index_dict = data_processor.feature_index_dict
        self.max_jets = data_processor.max_jets
        self.max_leptons = data_processor.max_leptons
        self.n_data_sets = n_folds * n_splits
        self.random_state = random_state
        self.model_list :list[BaseModel] = []
        self.k_fold = data_processor.create_k_folds(n_folds=n_folds, n_splits=n_splits, random_state=random_state)
        for i in range(self.n_data_sets):
            self.model_list.append(model_type(data_processor))
        def load_data_thread(model : BaseModel, data):
            model.load_data(*data)

        threads = []
        for i in range(self.n_data_sets):
            thread = threading.Thread(target=load_data_thread, args=(self.model_list[i], self.k_fold[i]))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def compute_sample_weights(self, **kwargs):
        def compute_weights_thread(model: BaseModel, alpha):
            model.compute_sample_weights(alpha=alpha)

        threads = []
        for model in self.model_list:
            thread = threading.Thread(target=compute_weights_thread, args=(model, ), kwargs=kwargs)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def build_models(self, **kwargs):
        def build_model_thread(model: BaseModel, **kwargs):
            model.build_model(**copy.deepcopy(kwargs))

        threads = []
        for model in self.model_list:
            thread = threading.Thread(target=build_model_thread, args=(model,), kwargs=kwargs)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()


    def compile_models(self, **kwargs):
        def compile_model_thread(model, **kwargs):
            model.compile_model(**copy.deepcopy(kwargs))

        threads = []
        for model in self.model_list:
            thread = threading.Thread(target=compile_model_thread, args=(model,), kwargs=kwargs)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def train_models(self, epochs=1, batch_size=1028, **kwargs):
        def train_model_thread(model : BaseModel, epochs, batch_size, **kwargs):
            model.train_model(epochs=epochs, batch_size=batch_size, **copy.deepcopy(kwargs))

        threads = []
        for model in self.model_list:
            thread = threading.Thread(target=train_model_thread, args=(model, epochs, batch_size), kwargs=kwargs)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

    def print_model_summary(self):
        self.model_list[0].summary()

    def enhance_region(self, variable, data_type, low_cut=None, high_cut=None, factor=1):
        for i in range(self.n_data_sets):
            self.model_list[i].enhance_region(variable, data_type, low_cut, high_cut, factor)

    def evaluate_accuracy(self):
        lep_1_accuracy = np.zeros(self.n_data_sets)
        lep_2_accuracy = np.zeros(self.n_data_sets)
        combined_accuracy = np.zeros(self.n_data_sets)

        def evaluate_accuracy_thread(model : BaseModel, index, lep_1_acc, lep_2_acc, combined_acc):
            lep_1_acc[index], lep_2_acc[index], combined_acc[index] = model.evaluate_accuracy()

        threads = []
        for i in range(self.n_data_sets):
            thread = threading.Thread(
            target=evaluate_accuracy_thread,
            args=(self.model_list[i], i, lep_1_accuracy, lep_2_accuracy, combined_accuracy)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        lep_1_accuracy_mean = np.mean(lep_1_accuracy)
        lep_2_accuracy_mean = np.mean(lep_2_accuracy)
        combined_accuracy_mean = np.mean(combined_accuracy)
        lep_1_accuracy_err = np.std(lep_1_accuracy) / np.sqrt(self.n_data_sets)
        lep_2_accuracy_err = np.std(lep_2_accuracy) / np.sqrt(self.n_data_sets)
        combined_accuracy_err = np.std(combined_accuracy) / np.sqrt(self.n_data_sets)
        return lep_1_accuracy_mean, lep_1_accuracy_err, lep_2_accuracy_mean, lep_2_accuracy_err, combined_accuracy_mean, combined_accuracy_err


    def accuracy_vs_feature(self, feature_name, data_type = "non_training", bins=50, xlims=None):
        if data_type not in self.feature_index_dict:
            raise ValueError(
                f"Data type {data_type} not found. Use 'jet', 'lepton', 'non_training', or 'global'."
            )
        if feature_name not in self.feature_index_dict[data_type]:
            raise ValueError(
                f"Feature {feature_name} not found in {data_type} features."
            )
        def compute_binned_accuracy_thread(model, index, feature_name, data_type, bins, xlims, results, feature_bins, feature_hist):
            results[index], feature_bins[index], feature_hist[index] = model.get_binned_accuracy(
            feature_name,
            data_type=data_type,
            bins=bins,
            xlims=xlims,
            )

        threads = []
        binned_accuracy_combined = [None] * self.n_data_sets
        feature_bins = [None] * self.n_data_sets
        feature_hist = [None] * self.n_data_sets
        for i in range(self.n_data_sets):
            thread = threading.Thread(
            target=compute_binned_accuracy_thread,
            args=(self.model_list[i], i, feature_name, data_type, bins, xlims, binned_accuracy_combined, feature_bins, feature_hist)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        binned_accuracy_combined = np.array(binned_accuracy_combined)
        feature_bins = feature_bins[0]
        feature_hist = feature_hist[0]

        binned_accuracy_combined = np.mean(binned_accuracy_combined, axis=0)
        binned_accuracy_combined_err = np.std(binned_accuracy_combined, axis=0) / np.sqrt(self.n_data_sets)
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        centers = (feature_bins[:-1] + feature_bins[1:]) / 2
        ax_twin = ax.twinx()
        ax_twin.set_ylabel("Feature Count", color='tab:orange')
        ax_twin.bar(
            centers,
            feature_hist,
            width=np.diff(feature_bins),
            alpha=0.3,
            color='tab:orange'
        )
        ax_twin.tick_params(axis="y", labelcolor='tab:orange')
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Accuracy")
        ax.tick_params(axis="y")
        ax.errorbar(
            centers,
            binned_accuracy_combined,
            yerr=binned_accuracy_combined_err,
            label=f"{self.model_type.__name__}",
            fmt='o',
            capsize=2,
        )
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Accuracy")
        ax.set_xlim(xlims)
        ax.set_ylim(0, 1.1)
        ax.legend()
        fig.tight_layout()
        return fig, ax
    
    def plot_permutation_importance(self, file_name=None):
        importance_scores_lep_1 = [None] * self.n_data_sets
        importance_scores_lep_2 = [None] * self.n_data_sets
        importance_scores_combined = [None] * self.n_data_sets

        def compute_importance_thread(model : BaseModel, index, results_lep_1, results_lep_2, results_combined):
            lep_1, lep_2, combined = model.compute_permutation_importance(shuffle_number=1)
            results_lep_1[index] = lep_1["mean"]
            results_lep_2[index] = lep_2["mean"]
            results_combined[index] = combined["mean"]

        threads = []
        for i in range(self.n_data_sets):
            thread = threading.Thread(
            target=compute_importance_thread,
            args=(self.model_list[i], i, importance_scores_lep_1, importance_scores_lep_2, importance_scores_combined)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()


        print(importance_scores_combined)
        importance_scores_lep_1_mean = pd.concat(importance_scores_lep_1, axis=1).mean(axis=1)
        importance_scores_lep_2_mean = pd.concat(importance_scores_lep_2, axis=1).mean(axis=1)
        importance_scores_combined_mean = pd.concat(importance_scores_combined, axis=1).mean(axis=1)

        importance_scores_lep_1_err = pd.concat(importance_scores_lep_1, axis=1).std(axis=1) / np.sqrt(self.n_data_sets)
        importance_scores_lep_2_err = pd.concat(importance_scores_lep_2, axis=1).std(axis=1) / np.sqrt(self.n_data_sets)
        importance_scores_combined_err = pd.concat(importance_scores_combined, axis=1).std(axis=1) / np.sqrt(self.n_data_sets)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        importance_scores_lep_1_mean = importance_scores_lep_1_mean.sort_values(ascending=False)
        importance_scores_lep_2_mean = importance_scores_lep_2_mean.sort_values(ascending=False)
        importance_scores_combined_mean = importance_scores_combined_mean.sort_values(ascending=False)
        importance_scores_lep_1_mean.plot(
            kind="bar",
            yerr=importance_scores_lep_1_err,
            ax=ax[0],
            color="blue",
            alpha=0.7,
        )
        importance_scores_lep_2_mean.plot(
            kind="bar",
            yerr=importance_scores_lep_2_err,
            ax=ax[1],
            color="orange",
            alpha=0.7,
        )
        importance_scores_combined_mean.plot(
            kind="bar",
            yerr=importance_scores_combined_err,
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
        return fig, ax

    def plot_confusion_matrix(self, data = None, labels = None, exclusive = True):
        confusion_matrices_lep_1 = [None] * self.n_data_sets
        confusion_matrices_lep_2 = [None] * self.n_data_sets
        max_jets = self.max_jets

        def get_confusion_matrix_thread(model : BaseModel, index, exclusive, results_lep_1, results_lep_2):
            confusion_matrix_lep_1, confusion_matrix_lep_2 = model.get_confusion_matrix(data = data, labels = labels, exclusive=exclusive)
            results_lep_1[index] = confusion_matrix_lep_1
            results_lep_2[index] = confusion_matrix_lep_2

        threads = []
        for i in range(self.n_data_sets):
            thread = threading.Thread(
            target=get_confusion_matrix_thread,
            args=(self.model_list[i], i, exclusive, confusion_matrices_lep_1, confusion_matrices_lep_2)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
        mean_confusion_lep_1 = np.mean(confusion_matrices_lep_1, axis=0)
        mean_confusion_lep_2 = np.mean(confusion_matrices_lep_2, axis=0)
        std_confusion_lep_1 = np.std(confusion_matrices_lep_1, axis=0) / np.sqrt(self.n_data_sets)
        std_confusion_lep_2 = np.std(confusion_matrices_lep_2, axis=0) / np.sqrt(self.n_data_sets)
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        sns.heatmap(mean_confusion_lep_1, annot=True, fmt='.2f', cmap='Blues', cbar=False, ax=ax[0])
        sns.heatmap(mean_confusion_lep_2, annot=True, fmt='.2f', cmap='Blues', cbar=False, ax=ax[1])
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


    def save_model(self, file_path="model.keras"):
        def save_model_thread(model, file_path):
            model.save_model(file_path)

        threads = []
        for i in range(self.n_data_sets):
            thread = threading.Thread(
            target=save_model_thread,
            args=(self.model_list[i], file_path.replace(".keras", f"_{i}.keras"))
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
        print(f"Models saved to {file_path}.")

    def load_model(self, file_path="model.keras"):
        for i in range(self.n_data_sets):
            file_path_i = file_path.replace(".keras", f"_{i}.keras")
            print("Loading model from", file_path_i)
            self.model_list[i].load_model(file_path_i)
        print(f"Models loaded from {file_path}.")

    def plot_history(self):
        fig, ax = [], []
        for i in range(self.n_data_sets):
            fig_i, ax_i = self.model_list[i].plot_history()
            fig.append(fig_i)
            ax.append(ax_i)
        return fig, ax
    
    def duplicate_jets(self):
        duplicate_jet_numbers = [None] * self.n_data_sets

        def duplicate_jets_thread(model, index, results):
            results[index] = model.duplicate_jets()

        threads = []
        for i in range(self.n_data_sets):
            thread = threading.Thread(
            target=duplicate_jets_thread,
            args=(self.model_list[i], i, duplicate_jet_numbers)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        duplicate_jet_numbers = np.array(duplicate_jet_numbers)
        return np.mean(duplicate_jet_numbers, axis=0), np.std(duplicate_jet_numbers, axis=0) / np.sqrt(self.n_data_sets)

    def plot_feature_variance(self, feature_name, data_type = "non_training", bins=50, xlims=None):
        if data_type not in self.feature_index_dict:
            raise ValueError(
                f"Data type {data_type} not found. Use 'jet', 'lepton', 'non_training', or 'global'."
            )
        if feature_name not in self.feature_index_dict[data_type]:
            raise ValueError(
                f"Feature {feature_name} not found in {data_type} features."
            )
        _ , hist_bins = np.histogram(
            np.concatenate([self.model_list[i].X_test[data_type][:, self.model_list[i].feature_index_dict[data_type][feature_name]] for i in range(self.n_data_sets)]),
            bins=bins,
            range=xlims,
        )
        feature_histos = np.zeros((self.n_data_sets, bins))
        for i in range(self.n_data_sets):  
            feature_histos[i], _ = np.histogram(
                self.model_list[i].X_test[data_type][:, self.model_list[i].feature_index_dict[data_type][feature_name]],
                bins=hist_bins,
            )
        feature_histos_min = np.min(feature_histos, axis=0)
        feature_histos_max = np.max(feature_histos, axis=0)
        feature_histos_mean = np.mean(feature_histos, axis=0)
        fig, ax = plt.subplots(figsize=(10, 5))
        centers = (hist_bins[:-1] + hist_bins[1:]) / 2
        ax.bar(
            centers,
            feature_histos_max - feature_histos_min,
            bottom=feature_histos_min,
            width=np.diff(hist_bins),
            alpha=0.5,
            color="orange",
            label="Feature Count Range",
        )
        for i in range(bins):
            ax.plot(
                [hist_bins[i], hist_bins[i + 1]],
                [feature_histos_mean[i], feature_histos_mean[i]],
                color="black",
            )
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Feature Count")
        ax.set_title("Feature Variance Across K-Folds")
        fig.tight_layout()
        return fig, ax