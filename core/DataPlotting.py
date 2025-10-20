from .DataLoader import DataPreprocessor, DataConfig
import numpy as np
import matplotlib.pyplot as plt
import os

class DataPlotter:
    def __init__(self, data_processor: DataPreprocessor, plots_dir: str):
        self.data_processor = data_processor
        self.plots_dir = plots_dir
        self.padding_value = data_processor.padding_value
        self.max_jets = data_processor.config.max_jets
        self.max_leptons = data_processor.config.max_leptons
        self.feature_index_dict = data_processor.feature_index_dict
        os.makedirs(self.plots_dir, exist_ok=True)

    def plot_feature_distribution(self, feature_type: str, feature_name: str, bins: int = 50):
        """
        Plots the distribution of a specified feature.

        Args:
            feature_type (str): The type of feature (e.g., 'jet', 'lepton', 'global').
            feature_name (str): The name of the feature to plot.
            bins (int): Number of bins for the histogram.
        """
        feature_data = self.data_processor.get_feature_data(feature_type, feature_name)
        plt.figure(figsize=(10, 6))
        plt.hist(feature_data.flatten()[feature_data.flatten() != self.padding_value], bins=bins, alpha=0.7, color='blue', edgecolor='black')
        plt.title(f'Distribution of {feature_name} ({feature_type})')
        plt.xlabel(feature_name)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, f'{feature_type}_{feature_name}_distribution.png'))
        plt.close()

    def plot_correlation_matrix(self):
        """
        Plots the correlation matrix of all features.
        """
        features_list = []
        feature_names = []


        # Collect all features
        for feature_type in ['jet', 'lepton', 'global']:
            for feature_name in self.data_processor.feature_index_dict[feature_type]:
                feature_data = self.data_processor.get_feature_data(feature_type, feature_name)
                if feature_data.shape[-1] > 1:
                    for i in range(feature_data.shape[-1]):
                        features_list.append(feature_data[..., i].flatten())
                        feature_names.append(f"{feature_name}_{i}")
                else:
                    features_list.append(feature_data.flatten())
                    feature_names.append(feature_name)


        # Compute correlation matrix
        correlation_matrix = np.zeros((len(features_list), len(features_list)))
        for i in range(len(features_list)):
            for j in range(len(features_list)):
                if i <= j:
                    valid_mask = (features_list[i] != self.padding_value) & (features_list[j] != self.padding_value)
                    if np.sum(valid_mask) > 1:
                        corr = np.corrcoef(features_list[i][valid_mask], features_list[j][valid_mask])[0, 1]
                    else:
                        corr = 0
                    correlation_matrix[i, j] = corr
                    correlation_matrix[j, i] = corr
        fig, ax = plt.subplots(figsize=(12, 10))
        cax = ax.matshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax)
        ax.set_xticks(np.arange(len(feature_names)))
        ax.set_yticks(np.arange(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=90)
        ax.set_yticklabels(feature_names)
        ax.set_title('Feature Correlation Matrix', pad=20)
        fig.tight_layout()
        fig.savefig(os.path.join(self.plots_dir, 'feature_correlation_matrix.png'))
        plt.close(fig)

    def plot_relational_jet_lepton_features(self, feature_function, name = "relational_feature", **kwargs):
        """
        Plots the value of a relational feature between jets and leptons.
        Args:
            feature_function (function): A function that takes jet and lepton feature arrays and computes a relational feature.
        """
        jet_features = self.data_processor.get_all_feature_data('jet')
        lepton_features = self.data_processor.get_all_feature_data('lepton')
        labels = self.data_processor.get_labels()
        num_events = jet_features.shape[0]

        matched_relational_features = []
        unmatched_relational_features = []

        for i in range(num_events):
            for j in range(self.max_jets):
                for k in range(self.max_leptons):
                    jet = jet_features[i, j]
                    lepton = lepton_features[i, k]
                    if (jet != self.padding_value).all() and (lepton != self.padding_value).all():
                        if labels[i,j,k] == 1:
                            matched_relational_features.append(feature_function(jet, lepton))
                        else:
                            unmatched_relational_features.append(feature_function(jet, lepton))
        fig, ax = plt.subplots(figsize=(10, 6))
        _, bins = np.histogram(matched_relational_features + unmatched_relational_features, **kwargs)
        ax.hist(matched_relational_features, bins=bins, alpha=0.7, label='Matched', color='green', edgecolor='black', density=True)
        ax.hist(unmatched_relational_features, bins=bins, alpha=0.7, label='Unmatched', color='red', edgecolor='black', density=True)
        ax.set_title('Relational Jet-Lepton Feature Distribution')
        ax.set_xlabel('Feature Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True)
        fig.savefig(os.path.join(self.plots_dir, f'{name}_distribution.png'))
        return fig, ax