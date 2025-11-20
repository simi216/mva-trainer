from .DataLoader import DataPreprocessor, DataConfig
import numpy as np
import matplotlib.pyplot as plt
import os


class DataPlotter:
    def __init__(self, data_processor: DataPreprocessor, plots_dir: str):
        self.data_processor = data_processor
        self.plots_dir = plots_dir
        self.padding_value = data_processor.data_config.padding_value
        self.max_jets = data_processor.data_config.max_jets
        self.NUM_LEPTONS = data_processor.data_config.NUM_LEPTONS
        self.feature_index_dict = data_processor.data_config.feature_indices
        self.event_cuts = np.ones(data_processor.data_length, dtype=bool)
        os.makedirs(self.plots_dir, exist_ok=True)

    def plot_feature_distribution(
        self, feature_type: str, feature_name: str, bins: int = 50
    ):
        """
        Plots the distribution of a specified feature.

        Args:
            feature_type (str): The type of feature (e.g., 'jet', 'lepton', 'met').
            feature_name (str): The name of the feature to plot.
            bins (int): Number of bins for the histogram.
        """
        feature_data = self.get_feature_data(feature_type, feature_name)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(
            feature_data.flatten()[feature_data.flatten() != self.padding_value],
            bins=bins,
            alpha=0.7,
            color="blue",
            edgecolor="black",
            density=True,
        )
        ax.set_title(f"Distribution of {feature_name} ({feature_type})")
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Frequency")
        ax.grid(True)
        return fig, ax

    def add_feature(self, function, name: str):
        """
        Adds a new feature to the data processor using a custom function.

        Args:
            function (function): A function that takes the data processor as input and returns the new feature array.
            name (str): The name of the new feature.
        """
        self.data_processor.add_custom_feature(function, name)

    def plot_correlation_matrix(self):
        """
        Plots the correlation matrix of all features.
        """
        features_list = []
        feature_names = []

        # Collect all features
        for feature_type in ["jet", "lepton", "met"]:
            for feature_name in self.data_processor.feature_index_dict[feature_type]:
                feature_data = self.get_feature_data(feature_type, feature_name)
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
                    valid_mask = (features_list[i] != self.padding_value) & (
                        features_list[j] != self.padding_value
                    )
                    if np.sum(valid_mask) > 1:
                        corr = np.corrcoef(
                            features_list[i][valid_mask], features_list[j][valid_mask]
                        )[0, 1]
                    else:
                        corr = 0
                    correlation_matrix[i, j] = corr
                    correlation_matrix[j, i] = corr
        fig, ax = plt.subplots(figsize=(12, 10))
        cax = ax.matshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(cax)
        ax.set_xticks(np.arange(len(feature_names)))
        ax.set_yticks(np.arange(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=90)
        ax.set_yticklabels(feature_names)
        ax.set_title("Feature Correlation Matrix", pad=20)
        fig.tight_layout()
        fig.savefig(os.path.join(self.plots_dir, "feature_correlation_matrix.png"))
        return fig, ax

    def register_data_cut(self, feature_type: str, feature_name: str, cut_function):
        """
        Registers a data cut based on a feature's value range.

        Args:
            feature_type (str): The type of feature (e.g., 'jet', 'lepton', 'met').
            feature_name (str): The name of the feature to apply the cut on.
            cut_function (function): A function that takes feature data and returns a boolean mask for valid events.
        """
        feature_data = self.get_feature_data(feature_type, feature_name)
        cut_mask = cut_function(feature_data)
        self.event_cuts = self.event_cuts & cut_mask

    def reset_data_cuts(self):
        """
        Resets all registered data cuts.
        """
        self.event_cuts = np.ones(self.data_processor.data_length, dtype=bool)

    def get_feature_data(self, feature_type: str, feature_name: str):
        """
        Retrieves feature data after applying registered data cuts.

        Args:
            feature_type (str): The type of feature (e.g., 'jet', 'lepton', 'met').
            feature_name (str): The name of the feature to retrieve.

        Returns:
            np.ndarray: The feature data after applying cuts.
        """
        feature_data = self.data_processor.get_feature_data(feature_type, feature_name)
        return feature_data[self.event_cuts]

    def get_all_feature_data(self, feature_type: str):
        """
        Retrieves all feature data of a specified type after applying registered data cuts.

        Args:
            feature_type (str): The type of feature (e.g., 'jet', 'lepton', 'met').

        Returns:
            np.ndarray: The feature data after applying cuts.
        """
        feature_data = self.data_processor.get_all_feature_data(feature_type)
        return feature_data[self.event_cuts]

    def plot_relational_jet_lepton_features(
        self, feature_function, name="relational_feature", **kwargs
    ):
        """
        Plots the value of a relational feature between jets and leptons.
        Args:
            feature_function (function): A function that takes jet and lepton feature arrays and computes a relational feature.
        """
        jet_features = self.get_all_feature_data("jet")
        lepton_features = self.get_all_feature_data("lepton")
        labels = self.get_all_feature_data("assignment_labels")


        lepton_extended = np.repeat(
            lepton_features[:, np.newaxis, :, :], self.max_jets, axis=1
        ) 
        jet_extended = np.repeat(
            jet_features[:, :, np.newaxis, :], self.NUM_LEPTONS, axis=2
        ) 
        relational_feature = feature_function(jet_extended.transpose(-1,1,2,0), lepton_extended.transpose(-1,1,2,0)).transpose(2,0,1)

        jet_mask = (jet_features[:, :, :] != self.padding_value ).any(axis=-1, keepdims=True)
        matched_relational_features = relational_feature[
            labels == 1 & jet_mask
        ].flatten()
        unmatched_relational_features = relational_feature[
            (labels == 0) & jet_mask
        ].flatten()

        fig, ax = plt.subplots(figsize=(10, 6))
        _, bins = np.histogram(
            np.concatenate((matched_relational_features, unmatched_relational_features)), **kwargs
        )
        ax.hist(
            matched_relational_features,
            bins=bins,
            alpha=0.7,
            label="Correct $b$-jet",
            color="tab:blue",
            density=True,
        )
        ax.hist(
            unmatched_relational_features,
            bins=bins,
            alpha=0.7,
            label="Other jets",
            color="tab:red",
            density=True,
        )
        ax.set_xlabel(f"{name}")
        ax.set_ylabel("Normalized Frequency")
        ax.legend()
        ax.grid(True)
        return fig, ax

    def plot_2d_feature_distribution(
        self,
        feature_type_x: str,
        feature_name_x: str,
        feature_type_y: str,
        feature_name_y: str,
        normalise="none",
        plot_average=False,
        **kwargs,
    ):
        """
        Plots a 2D histogram of two specified features.
        Args:
            feature_type_x (str): The type of the x-axis feature (e.g., 'jet', 'lepton', 'met').
            feature_name_x (str): The name of the x-axis feature to plot.
            feature_type_y (str): The type of the y-axis feature (e.g., 'jet', 'lepton', 'met').
            feature_name_y (str): The name of the y-axis feature to plot.
            normalise (str): Normalisation method for the histogram ('none', 'x', 'y', 'all').
            plot_average (bool): Whether to plot the average of the y feature in each x bin
            kwargs: Additional keyword arguments for np.hist2d to customise the binning and appearance.
        """
        feature_data_x = self.get_feature_data(feature_type_x, feature_name_x)
        feature_data_y = self.get_feature_data(feature_type_y, feature_name_y)
        hist2d, bins_x, bins_y = np.histogram2d(
            feature_data_x.flatten()[feature_data_x.flatten() != self.padding_value],
            feature_data_y.flatten()[feature_data_y.flatten() != self.padding_value],
            **kwargs,
        )
        # Normalisation
        if normalise == "x":
            hist2d = hist2d / hist2d.sum(axis=1, keepdims=True)
        elif normalise == "y":
            hist2d = hist2d / hist2d.sum(axis=0, keepdims=True)
        elif normalise == "all":
            hist2d = hist2d / hist2d.sum()

        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.imshow(
            hist2d.T,
            origin="lower",
            aspect="auto",
            extent=[bins_x[0], bins_x[-1], bins_y[0], bins_y[-1]],
            cmap="viridis",
        )
        if plot_average:
            # Get average of the y feature in each x bin
            bin_centers_x = 0.5 * (bins_x[:-1] + bins_x[1:])
            bin_centers_y = 0.5 * (bins_y[:-1] + bins_y[1:])
            avg_y = []
            for i in range(len(bins_x) - 1):
                mask = (
                    (feature_data_x.flatten() >= bins_x[i])
                    & (feature_data_x.flatten() < bins_x[i + 1])
                    & (feature_data_y.flatten() != self.padding_value)
                )
                if np.sum(mask) > 0:
                    avg_y.append(np.mean(feature_data_y.flatten()[mask]))
                else:
                    avg_y.append(np.nan)
            ax.plot(
                bin_centers_x,
                avg_y,
                color="red",
                marker="o",
                linestyle="-",
                label=f"Average {feature_name_y}",
            )
            ax.legend()
        fig.colorbar(cax, ax=ax)
        ax.set_xlabel(f"{feature_name_x}")
        ax.set_ylabel(f"{feature_name_y} ")
        ax.set_title(f"2D Distribution of {feature_name_x} and {feature_name_y}")
        return fig, ax
