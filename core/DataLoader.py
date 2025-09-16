import uproot
import awkward as ak
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import convert_to_tensor


class DataLoader:
    """
    DataLoader is a utility class for loading and preprocessing data from ROOT files using uproot and awkward arrays.
    It supports feature selection, padding, and clipping for use in machine learning workflows.
    Attributes:
        features (list): List of feature names to be loaded from the ROOT file, derived from the keys of feature_clipping.
        clipping (dict): Dictionary mapping feature names to their clipping (padding) values.
        data (pd.DataFrame or None): Loaded and processed data as a pandas DataFrame, or None if not loaded.
    Methods:
        __init__(feature_clipping):
            Initializes the DataLoader with a dictionary specifying feature names and their clipping values.
        load_data(file_path, tree_name, max_events=None):
            Loads data from a ROOT file and tree, selecting only the specified features and applying clipping.
            Args:
                file_path (str): Path to the ROOT file.
                tree_name (str): Name of the tree within the ROOT file.
                max_events (int, optional): Maximum number of events to load. Loads all if None.
            Raises:
                ValueError: If file_path is not a string, tree_name is not found, or required branches are missing.
        _get_padded_data(padding_value=-999):
            Internal method to pad and clip features according to the clipping dictionary.
            Pads missing values with the specified padding_value and expands features as needed.
            Args:
                padding_value (numeric, optional): Value to use for padding missing entries. Default is -999.
            Raises:
                ValueError: If a specified feature is not found in the loaded data.
        get_data():
            Returns the processed data as a pandas DataFrame.
            Returns:
                pd.DataFrame: The loaded and processed data.
            Raises:
                ValueError: If data has not been loaded yet.
    """

    def __init__(self, feature_clipping):
        """
        Initializes the data loader with feature clipping information.
        Args:
            feature_clipping (dict): A dictionary where keys are feature names (str) and values are the corresponding clipping values.
        Raises:
            ValueError: If feature_clipping is not a dictionary.
        """

        self.features = list(feature_clipping.keys())
        self.clipping = feature_clipping
        self.data = None
        if not isinstance(self.clipping, dict):
            raise ValueError(
                "Clipping should be a dictionary with feature names as keys and clipping values as values."
            )

    def load_data(self, file_path, tree_name, max_events=None):
        """
        Loads data from a ROOT file using uproot, extracting specified branches from a given tree.
        Parameters
        ----------
        file_path : str
            Path to the ROOT file to be loaded.
        tree_name : str
            Name of the tree within the ROOT file from which to extract data.
        max_events : int, optional
            Maximum number of events to load from the tree. If None, all events are loaded.
        Raises
        ------
        ValueError
            If `file_path` is not a string.
            If `tree_name` is not found in the file.
            If any of the required branches are missing in the tree.
            If no data is found for the specified tree and branches.
        Side Effects
        ------------
        Sets `self.data` to the loaded awkward array.
        Calls `self._get_padded_data()` to process the loaded data.
        """

        branches = self.features
        if not isinstance(file_path, str):
            raise ValueError("File path should be a string.")
        with uproot.open(file_path) as file:
            if tree_name not in file:
                raise ValueError(f"Tree {tree_name} not found in file {file_path}.")
            missing_branches = [
                branch for branch in branches if branch not in file[tree_name].keys()
            ]
            if missing_branches:
                raise ValueError(
                    f"The following branches are missing in the tree {tree_name}: {missing_branches}"
                )
            if max_events is not None:
                data = file[tree_name].arrays(
                    branches, library="ak", entry_stop=max_events
                )
            else:
                data = file[tree_name].arrays(branches, library="ak")
        if data is None:
            raise ValueError(
                f"No data found in {file_path} for tree {tree_name} with branches {branches}"
            )
        self.data = data
        self._get_padded_data()

    def _get_padded_data(self, padding_value=-999):
        """
        Pads and processes feature data according to the specified clipping configuration.
        For each feature in `self.features`, this method checks if the feature exists in the data.
        If the feature's clipping value is 1, missing values are filled with `padding_value`.
        If the clipping value is greater than 1, the feature is padded or clipped to the specified length,
        missing values are filled with `padding_value`, and the resulting array is split into separate columns
        for each padded element (e.g., `feature_0`, `feature_1`, ...).
        The processed data is stored as a pandas DataFrame in `self.data`.
        Args:
            padding_value (int, optional): The value to use for padding missing data. Defaults to -999.
        Raises:
            ValueError: If a feature in `self.features` is not found in the data.
        """

        data_dict = {}
        for feature in self.features:
            if feature not in self.data.fields:
                raise ValueError(f"Feature {feature} not found in data.")
            if self.clipping[feature] == 1:
                data_dict[feature] = ak.fill_none(self.data[feature], padding_value)
                continue
            else:
                feature_variable = ak.fill_none(
                    ak.pad_none(
                        self.data[feature], self.clipping[feature], clip=True, axis=1
                    ),
                    padding_value,
                )
                for i in range(self.clipping[feature]):
                    data_dict[f"{feature}_{i}"] = feature_variable[:, i]
        self.data = pd.DataFrame(data_dict)

    def get_data(self):
        """
        Retrieve the loaded data.
        Returns:
            Any: The data that has been loaded.
        Raises:
            ValueError: If the data has not been loaded prior to calling this method.
        """

        if self.data is None:
            raise ValueError(
                "Data not loaded. Please load data using load_data() method."
            )
        return self.data


class DataPreprocessor:
    """
    A class for preprocessing and managing data for machine learning tasks involving jets, leptons, and global features.
    Attributes:
        jet_features (list[str]): List of jet feature names.
        lepton_features (list[str]): List of lepton feature names.
        jet_truth_label (str): Name of the jet truth label.
        lepton_truth_label (str): Name of the lepton truth label.
        max_leptons (int): Maximum number of leptons per event. Default is 2.
        max_jets (int): Maximum number of jets per event. Default is 4.
        global_features (list[str]): List of global feature names. Default is None.
        non_training_features (list[str]): List of features not used for training. Default is None.
        regression_targets (list[str]): List of regression target names. Default is None.
        event_weight (str): Name of the event weight feature. Default is None.
        padding_value (float): Value used for padding missing data. Default is -999.0.
    Methods:
        apply_cut(cut_feature, cut_low=None, cut_high=None):
            Apply a cut on a specific feature based on lower and/or upper bounds.
        reorder_by_feature(reorder_feature):
            Reorder jets in each event based on a specific feature.
        load_data(file_path, tree_name, max_events=None, cut_neg_weights=True):
            Load data from a file and apply cuts if specified.
        prepare_data():
            Prepare data by computing pairwise features and building feature pairs.
        build_pairs():
            Build feature pairs for jets and leptons, and organize data into structured arrays.
        get_event_weight(cut_neg_weights=False):
            Retrieve and normalize event weights.
        build_labels():
            Build labels for training based on truth information.
        split_data(test_size=0.2, random_state=42):
            Split data into training and testing sets.
        get_data():
            Retrieve training and testing data.
        create_k_folds(n_folds=5, n_splits=1, random_state=42):
            Create k-folds for cross-validation.
        plot_feature_correlation():
            Plot the correlation matrix of features.
        enhance_feature_value(feature_name, enhancement_value):
            Enhance the value of a specific feature based on labels.
        get_labels():
            Retrieve the labels for the data.
        get_feature_data(dataType, feature_name):
            Retrieve specific feature data based on its type and name.
        normalise_data():
            Normalize the feature data by removing padding and scaling.
        plot_feature_distribution(data, feature_name, file_name=None, **kwargs):
            Plot the distribution of a specific feature.
    Raises:
        ValueError: Raised in various methods when required data is missing or invalid arguments are provided.
    """

    def __init__(
        self,
        jet_features: list[str],
        lepton_features: list[str],
        jet_truth_label: str,
        lepton_truth_label: str,
        max_leptons: int = 2,
        max_jets: int = 4,
        global_features: list[str] = None,
        non_training_features: list[str] = None,
        regression_targets: list[str] = None,
        event_weight: str = None,
    ):
        """
        Initializes the DataLoader class with the specified features, labels, and configuration.
        Args:
            jet_features (list[str]): List of features associated with jets.
            lepton_features (list[str]): List of features associated with leptons.
            jet_truth_label (str): Label for the truth information of jets.
            lepton_truth_label (str): Label for the truth information of leptons.
            max_leptons (int, optional): Maximum number of leptons to consider. Defaults to 2.
            max_jets (int, optional): Maximum number of jets to consider. Defaults to 4.
            global_features (list[str], optional): List of global features for the event. Defaults to None.
            non_training_features (list[str], optional): List of features excluded from training. Defaults to None.
            regression_targets (list[str], optional): List of regression target labels. Defaults to None.
            event_weight (str, optional): Label for the event weight. Defaults to None.
        Attributes:
            jet_features (list[str]): Stores the jet features.
            lepton_features (list[str]): Stores the lepton features.
            jet_truth_label (str): Stores the jet truth label.
            lepton_truth_label (str): Stores the lepton truth label.
            max_leptons (int): Maximum number of leptons.
            max_jets (int): Maximum number of jets.
            global_features (list[str]): Stores the global features.
            n_jets (int): Number of jet features.
            n_leptons (int): Number of lepton features.
            n_global (int): Number of global features.
            data (Any): Placeholder for the loaded data.
            data_length (int): Length of the loaded data.
            padding_value (float): Value used for padding missing data. Defaults to -999.0.
            load_jet_features (list[str]): Stores the jet features for loading.
            load_lepton_features (list[str]): Stores the lepton features for loading.
            feature_index_dict (dict): Dictionary mapping feature names to indices.
            labels (Any): Placeholder for labels.
            feature_data (Any): Placeholder for feature data.
            non_training_features (list[str]): Stores the non-training features.
            n_non_training (int): Number of non-training features.
            event_weight_label (str): Stores the event weight label.
            X_train (Any): Placeholder for training feature data.
            X_test (Any): Placeholder for testing feature data.
            y_train (Any): Placeholder for training labels.
            y_test (Any): Placeholder for testing labels.
            regression_targets (list[str]): Stores the regression target labels.
            n_regression_targets (int): Number of regression targets.
            cut_dict (dict): Dictionary for storing cut conditions.
            data_normalisation_factors (dict): Dictionary for storing data normalization factors.
        """

        self.jet_features = jet_features
        self.lepton_features = lepton_features
        self.jet_truth_label = jet_truth_label
        self.lepton_truth_label = lepton_truth_label
        self.max_leptons = max_leptons
        self.max_jets = max_jets
        self.global_features = global_features
        self.n_jets: int = len(jet_features)
        self.n_leptons: int = len(lepton_features)
        self.n_global: int = len(global_features) if global_features else 0
        self.data = None
        self.data_length = None
        self.padding_value = -999.0
        self.load_jet_features = jet_features
        self.load_lepton_features = lepton_features
        self.feature_index_dict = {}
        self.labels = None
        self.feature_data = None
        self.non_training_features = non_training_features
        self.n_non_training = len(non_training_features) if non_training_features else 0
        self.event_weight_label = event_weight
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.regression_targets = regression_targets
        self.n_regression_targets = len(regression_targets) if regression_targets else 0
        self.cut_dict = {}
        self.data_normalisation_factors = {}

    def apply_cut(self, cut_feature, cut_low=None, cut_high=None):
        """
        Apply a cut to a specific feature by specifying lower and/or upper bounds.
        Parameters:
            cut_feature (str): The name of the feature to which the cut will be applied.
            cut_low (float, optional): The lower bound of the cut. If None, no lower bound is applied. Default is None.
            cut_high (float, optional): The upper bound of the cut. If None, no upper bound is applied. Default is None.
        Raises:
            ValueError: If both `cut_low` and `cut_high` are None.
            ValueError: If `cut_low` is not None, `cut_high` is not None, and `cut_low` is greater than or equal to `cut_high`.
        Updates:
            self.cut_dict: Adds or updates an entry for `cut_feature` with the specified bounds as a tuple (cut_low, cut_high).
        """

        if cut_low is None and cut_high is None:
            raise ValueError("At least one of cut_low or cut_high must be specified.")
        if cut_low is not None and cut_high is not None and cut_low >= cut_high:
            raise ValueError("cut_low must be less than cut_high.")
        self.cut_dict[cut_feature] = (cut_low, cut_high)


    def reorder_by_feature(self, reorder_feature):
        """
        Reorders the feature data and corresponding labels for each event based on the values of a specified feature.
        This method sorts the jets in each event according to the values of the specified feature. Jets with masked
        values (equal to the padding value) are not reordered and remain in their original positions.
        Args:
            reorder_feature (str): The name of the feature to use for reordering the jets.
        Raises:
            ValueError: If the feature data is not prepared (i.e., `self.feature_data` is None).
            ValueError: If the specified feature is not found in the feature index dictionary (`self.feature_index_dict`).
            ValueError: If the labels are not prepared (i.e., `self.labels` is None).
        Notes:
            - The method assumes that `self.feature_data` is a 3D numpy array with shape
              (number_of_events, number_of_jets, number_of_features).
            - The method assumes that `self.labels` is a 2D numpy array with shape
              (number_of_events, number_of_jets).
            - The `self.padding_value` is used to identify masked jets that should not be reordered.
        """

        if self.feature_data is None:
            raise ValueError(
                "Feature data not prepared. Please prepare data using prepare_data() method."
            )
        if reorder_feature not in self.feature_index_dict:
            raise ValueError(
                f"Feature {reorder_feature} not found in feature index dictionary."
            )
        if self.labels is None:
            raise ValueError(
                "Labels not prepared. Please prepare data using prepare_data() method."
            )
        feature_index = self.feature_index_dict[reorder_feature]
        for event_index in range(self.feature_data.shape[0]):
            # Extract the feature values for the jets in the current event
            feature_values = self.feature_data[event_index, :, feature_index]
            # Identify unmasked jets (those not equal to the padding value)
            unmasked_indices = np.where(feature_values != self.padding_value)[0]
            # Get the sorting indices for unmasked jets based on the feature values
            sorted_indices = unmasked_indices[
                np.argsort(feature_values[unmasked_indices])
            ]
            # Create a new ordering array that keeps masked jets in place
            full_sorted_indices = np.arange(self.feature_data.shape[1])
            full_sorted_indices[unmasked_indices] = sorted_indices

            # Reorder the feature data and labels based on the sorted indices
            self.feature_data[event_index] = self.feature_data[
                event_index, full_sorted_indices
            ]
            self.labels[event_index] = self.labels[event_index, full_sorted_indices]


    def load_data(self, file_path, tree_name, max_events=None, cut_neg_weights=True):
        """
        Load data from a specified file and apply preprocessing steps.
        Parameters:
        -----------
        file_path : str
            Path to the input file containing the data.
        tree_name : str
            Name of the tree structure within the file to load data from.
        max_events : int, optional
            Maximum number of events to load. If None, all events are loaded. Default is None.
        cut_neg_weights : bool, optional
            If True, filters out events with negative weights based on the event weight label. Default is True.
        Raises:
        -------
        ValueError
            If a cut feature specified in `cut_dict` is not found in the data.
            If data is already loaded and the method is called again.
        Notes:
        ------
        - The method initializes a DataLoader instance with feature clipping based on the class attributes.
        - Applies cuts specified in `cut_dict` to filter the data.
        - Filters out events with negative weights if `cut_neg_weights` is True.
        - Stores the processed data in the `self.data` attribute and its length in `self.data_length`.
        - Clears the DataLoader instance after loading data to save memory.
        """

        if self.data is None:
            feature_clipping = {
                feature: self.max_jets for feature in self.load_jet_features
            }
            feature_clipping.update(
                {feature: self.max_leptons for feature in self.load_lepton_features}
            )
            (
                feature_clipping.update(
                    {feature: 1 for feature in self.global_features}
                )
                if self.global_features
                else None
            )
            feature_clipping.update({self.jet_truth_label: 6})
            feature_clipping.update({self.lepton_truth_label: 2})
            (
                feature_clipping.update(
                    {feature: 1 for feature in self.non_training_features}
                )
                if self.non_training_features
                else None
            )
            (
                feature_clipping.update({self.event_weight_label: 1})
                if self.event_weight_label
                else None
            )
            (
                feature_clipping.update(
                    {
                        regression_target: 1
                        for regression_target in self.regression_targets
                    }
                )
                if self.regression_targets
                else None
            )
            DataHandle = DataLoader(feature_clipping)
            DataHandle.load_data(file_path, tree_name, max_events=max_events)
            self.data = DataHandle.get_data()
            DataHandle = None  # Clear the DataLoader instance to save memory
            if self.cut_dict:
                for cut_feature, (cut_low, cut_high) in self.cut_dict.items():
                    if cut_feature not in self.data.columns:
                        raise ValueError(
                            f"Cut feature {cut_feature} not found in data."
                        )
                    print(
                        f"Applying cut on feature {cut_feature}: low={cut_low}, high={cut_high}"
                    )
                    print(
                        f"{self.data[cut_feature].min()} <= {cut_feature} <= {self.data[cut_feature].max()}"
                    )
                    if cut_low is not None:
                        self.data = self.data[self.data[cut_feature] >= cut_low]
                    if cut_high is not None:
                        self.data = self.data[self.data[cut_feature] <= cut_high]
            if self.event_weight_label is not None and cut_neg_weights:
                self.data = self.data[self.data[self.event_weight_label] >= 0]
            self.data_length = len(self.data)
        else:
            raise ValueError(
                "Data already loaded. Please use a different instance of the class to load new data."
            )

    def prepare_data(self):
        """
        Prepares the data for further processing by computing pairwise features
        and building pairs.
        This method performs the following steps:
        1. Checks if the data is loaded. If not, raises a ValueError.
        2. Builds pairs from the data.
        Raises:
            ValueError: If the data has not been loaded prior to calling this method.
        """

        if self.data is None:
            raise ValueError(
                "Data not loaded. Please load data using load_data() method."
            )
        self.build_pairs()
        # self.reorder_by_feature("dR_lep_jet")

    def build_pairs(self):
        """
        Builds and organizes feature data into structured arrays for training and evaluation.
        This method processes the loaded data to create structured arrays for different feature types
        (lepton, jet, global, non-training, event weight, and regression targets). It also updates
        feature index dictionaries to map feature names to their respective indices.
        Raises:
            ValueError: If the data has not been loaded prior to calling this method.
        Processes:
            - Lepton features: Extracts and reshapes lepton-related data.
            - Jet features: Extracts and reshapes jet-related data
            - Global features: Extracts global features if available.
            - Non-training features: Extracts non-training features if available.
            - Event weight: Extracts event weight data if specified.
            - Regression targets: Extracts regression target data if specified.
        Updates:
            - `self.feature_index_dict`: A dictionary mapping feature categories (e.g., "lepton", "jet")
              to their respective feature indices.
            - `self.feature_data`: A dictionary containing the processed feature data arrays.
        Notes:
            - Clears the original data (`self.data`) after processing to save memory.
            - Calls `self.build_labels()` to further process labels after building feature data.
        """

        if self.data is None:
            raise ValueError(
                "Data not loaded. Please load data using load_data() method."
            )

        lepton_data = (
            self.data[
                [
                    lepton_var + f"_{lep_index}"
                    for lepton_var in self.lepton_features
                    for lep_index in range(self.max_leptons)
                ]
            ]
            .to_numpy()
            .reshape(self.data_length, -1, self.max_leptons)
            .transpose((0, 2, 1))
        )
        lepton_indices = {
            lepton_var: idx for idx, lepton_var in enumerate(self.lepton_features)
        }

        jet_data = (
            self.data[
                [
                    jet_var + f"_{jet_index}"
                    for jet_var in self.jet_features
                    for jet_index in range(self.max_jets)
                ]
            ]
            .to_numpy()
            .reshape(self.data_length, -1, self.max_jets)
            .transpose((0, 2, 1))
        )
        jet_indices = {
            jet_var: idx for idx, jet_var in enumerate(self.jet_features)
        }

        global_data = (
            self.data[[global_var for global_var in self.global_features]]
            .to_numpy()
            .reshape(self.data_length, 1, self.n_global)
            if self.global_features
            else None
        )
        global_indices = (
            {global_var: idx for idx, global_var in enumerate(self.global_features)}
            if self.global_features
            else {}
        )
        non_training_data = (
            self.data[
                [non_training_var for non_training_var in self.non_training_features]
            ].to_numpy()
            if self.non_training_features
            else None
        )
        non_training_indices = (
            {
                non_training_var: idx
                for idx, non_training_var in enumerate(self.non_training_features)
            }
            if self.non_training_features
            else {}
        )

        feature_dict = {}

        self.feature_index_dict.update({"lepton": lepton_indices})
        feature_dict.update({"lepton": lepton_data})
        self.feature_index_dict.update({"jet": jet_indices})
        feature_dict.update({"jet": jet_data})
        if self.global_features is not None:
            self.feature_index_dict.update({"global": global_indices})
            feature_dict.update({"global": global_data})
        if self.non_training_features is not None:
            self.feature_index_dict.update({"non_training": non_training_indices})
            feature_dict.update({"non_training": non_training_data})
        if self.event_weight_label is not None:
            feature_dict.update(
                {"event_weight": self.data[self.event_weight_label].to_numpy()}
            )
            self.feature_index_dict.update(
                {"event_weight": {self.event_weight_label: 0}}
            )
        if self.regression_targets is not None:
            feature_dict.update(
                {"regression_targets": self.data[self.regression_targets].to_numpy()}
            )
            self.feature_index_dict.update(
                {
                    "regression_targets": {
                        regression_target: idx
                        for idx, regression_target in enumerate(self.regression_targets)
                    }
                }
            )
        self.feature_data = feature_dict

        self.build_labels()
        self.data = None  # Clear the original data to save memory

    def get_event_weight(self, cut_neg_weights=False):
        """
        Calculate and return normalized event weights.
        This method retrieves the event weights from the feature data, optionally
        filters out negative weights, normalizes the weights so that their sum equals 1,
        and updates the data length attribute.
        Args:
            cut_neg_weights (bool, optional): If True, negative weights are excluded
                from the calculation. Defaults to False.
        Returns:
            numpy.ndarray: A normalized array of event weights.
        Raises:
            ValueError: If feature data is not loaded or the event weight label is not provided.
        """

        if self.feature_data is None:
            raise ValueError(
                "Feature Data not loaded. Please load data using load_data() method."
            )
        if self.event_weight_label is None:
            raise ValueError("Event weight label not provided.")
        event_weight = self.feature_data["event_weight"]
        if cut_neg_weights:
            event_weight = event_weight[event_weight >= 0]
        event_weight = event_weight / np.sum(event_weight)
        self.data_length = len(event_weight)

        return event_weight

    def build_labels(self):
        """
        Builds and processes labels for the dataset based on jet and lepton truth information.
        This method performs the following steps:
        1. Validates that data has been loaded; raises a ValueError if not.
        2. Extracts jet and lepton truth information from the dataset.
        3. Applies a reconstruction success mask to filter out invalid or out-of-range data.
        4. Updates feature data to only include entries that pass the reconstruction success mask.
        5. Constructs a pair truth tensor that maps jet-lepton pairings for each event.
        6. Sets the labels for the dataset based on regression targets if available, otherwise uses the pair truth tensor.
        Attributes Updated:
        - `self.feature_data`: Filters feature data based on the reconstruction success mask.
        - `self.data_length`: Updates the length of the filtered dataset.
        - `self.labels`: Sets the labels for the dataset.
        - `self.feature_data["labels"]`: Stores the pair truth tensor as part of the feature data.
        Raises:
        - ValueError: If the data has not been loaded prior to calling this method.
        Notes:
        - The method assumes that `self.data` contains the necessary columns for jet and lepton truth labels.
        - The pair truth tensor is a 3D array of shape (data_length, max_jets, max_leptons), where each entry
          indicates whether a specific jet-lepton pairing is valid for a given event.
        """

        if self.data is None:
            raise ValueError(
                "Data not loaded. Please load data using load_data() method."
            )

        jet_truth = self.data[
            [self.jet_truth_label + f"_{0}", self.jet_truth_label + f"_{3}"]
        ].to_numpy()
        lepton_truth = self.data[
            [self.lepton_truth_label + f"_{0}", self.lepton_truth_label + f"_{1}"]
        ].to_numpy()

        reco_success_mask = (jet_truth != -1).all(axis=1) & (
            jet_truth < self.max_jets
        ).all(axis=1)

        for data in self.feature_data:
            if self.feature_data[data] is not None:
                self.feature_data[data] = self.feature_data[data][reco_success_mask]
        self.data_length = len(jet_truth[reco_success_mask])
        jet_truth = jet_truth[reco_success_mask]
        lepton_truth = lepton_truth[reco_success_mask]

        pair_truth = np.zeros((self.data_length, self.max_jets, self.max_leptons))
        for event_index in range(self.data_length):
            for lep_index in range(self.max_leptons):
                for jet_index in range(self.max_jets):
                    if (
                        jet_truth[event_index, 0] == jet_index
                        and lepton_truth[event_index, 0] == lep_index
                    ):
                        pair_truth[event_index, jet_index, lep_index] = 1
                    elif (
                        jet_truth[event_index, 1] == jet_index
                        and lepton_truth[event_index, 1] == lep_index
                    ):
                        pair_truth[event_index, jet_index, lep_index] = 1
                    else:
                        pair_truth[event_index, jet_index, lep_index] = 0

        if self.regression_targets is not None:
            self.labels = self.feature_data["regression_targets"]
        else:
            self.labels = pair_truth
        self.feature_data["labels"] = pair_truth

    def split_data(self, test_size=0.2, random_state=42):
        """
        Splits the feature data and labels into training and testing sets.
        This method uses scikit-learn's `train_test_split` to divide the data into
        training and testing subsets. The split is performed for each key in the
        `feature_data` dictionary. The resulting training and testing data are stored
        in the `X_train`, `X_test`, `y_train`, and `y_test` attributes.
        Parameters:
            test_size (float, optional): Proportion of the dataset to include in the test split.
                                         Default is 0.2 (20% test data).
            random_state (int, optional): Random seed for reproducibility of the split.
                                          Default is 42.
        Raises:
            ValueError: If `feature_data` is not prepared (i.e., is None).
            ValueError: If `labels` are not prepared (i.e., is None).
        Attributes:
            X_train (dict): Dictionary containing training feature data for each key in `feature_data`.
            X_test (dict): Dictionary containing testing feature data for each key in `feature_data`.
            y_train (array-like): Training labels.
            y_test (array-like): Testing labels.
        """

        if self.feature_data is None:
            raise ValueError(
                "Feature data not prepared. Please prepare data using prepare_data() method."
            )
        if self.labels is None:
            raise ValueError(
                "Labels not prepared. Please prepare data using prepare_data() method."
            )
        from sklearn.model_selection import train_test_split

        X_train = {}
        X_test = {}
        y_train = None
        y_test = None
        for data in self.feature_data:
            if self.feature_data[data] is not None:
                X_train[data], X_test[data], y_train, y_test = train_test_split(
                    self.feature_data[data],
                    self.labels,
                    test_size=test_size,
                    random_state=random_state,
                )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def get_data(self):
        """
        Retrieves the training and testing datasets.
        This method ensures that the data is split into training and testing sets
        before returning them. If the data has not been split yet, it calls the
        `split_data` method to perform the split.
        Returns:
            tuple: A tuple containing four elements:
                - X_train (array-like): Features for the training set.
                - y_train (array-like): Labels for the training set.
                - X_test (array-like): Features for the testing set.
                - y_test (array-like): Labels for the testing set.
        """

        if self.X_train is None or self.X_test is None:
            self.split_data()
        return self.X_train, self.y_train, self.X_test, self.y_test

    def create_k_folds(
        self, n_folds: int = 5, n_splits: int = 1, random_state: int = 42
    ) -> list[
        tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray]
    ]:
        """
        Creates k-fold cross-validation splits for the dataset.
        This method divides the dataset into `n_splits` subsets, and for each subset,
        performs k-fold cross-validation by splitting the data into training and testing sets.
        The data is shuffled before splitting to ensure randomness.
        Args:
            n_folds (int, optional): The number of folds for cross-validation. Defaults to 5.
            n_splits (int, optional): The number of splits of the dataset. Defaults to 1.
            random_state (int, optional): Seed for random number generator to ensure reproducibility. Defaults to 42.
        Returns:
            list[tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray]]:
                A list of tuples, where each tuple contains:
                    - X_train (dict[str, np.ndarray]): Training feature data for the fold.
                    - y_train (np.ndarray): Training labels for the fold.
                    - X_test (dict[str, np.ndarray]): Testing feature data for the fold.
                    - y_test (np.ndarray): Testing labels for the fold.
        Raises:
            ValueError: If `feature_data` is not prepared (i.e., None).
            ValueError: If `labels` are not prepared (i.e., None).
        Notes:
            - The method modifies `self.feature_data` and `self.labels` in-place by shuffling them.
            - The last fold or split may contain more samples if the data length is not evenly divisible
              by `n_folds` or `n_splits`.
        """

        if self.feature_data is None:
            raise ValueError("Feature data not prepared.")
        if self.labels is None:
            raise ValueError("Labels not prepared.")

        if random_state is not None:
            np.random.seed(random_state)

        # Shuffle once
        indices = np.arange(self.data_length)
        np.random.shuffle(indices)

        # Shuffle all data in-place (so views remain valid)
        for key in self.feature_data:
            if self.feature_data[key] is not None:
                self.feature_data[key] = self.feature_data[key][indices]
        self.labels = self.labels[indices]

        split_size = self.data_length // n_splits
        folded_data = []

        for split_index in range(n_splits):
            start = split_index * split_size
            end = (
                (split_index + 1) * split_size
                if split_index < n_splits - 1
                else self.data_length
            )

            split_len = end - start
            fold_size = split_len // n_folds
            for fold_index in range(n_folds):
                test_start = start + fold_index * fold_size
                test_end = test_start + fold_size if fold_index < n_folds - 1 else end

                train1_start = start
                train1_end = test_start
                train2_start = test_end
                train2_end = end

                X_train = {
                    k: np.concatenate(
                        [
                            self.feature_data[k][train1_start:train1_end],
                            self.feature_data[k][train2_start:train2_end],
                        ]
                    )
                    for k in self.feature_data
                }
                y_train = np.concatenate(
                    [
                        self.labels[train1_start:train1_end],
                        self.labels[train2_start:train2_end],
                    ]
                )
                X_test = {
                    k: self.feature_data[k][test_start:test_end]
                    for k in self.feature_data
                }
                y_test = self.labels[test_start:test_end]

                folded_data.append((X_train, y_train, X_test, y_test))

        return folded_data

    def plot_feature_correlation(self):
        if self.feature_data is None:
            raise ValueError(
                "Feature data not prepared. Please prepare data using prepare_data() method."
            )
        feature_data = pd.DataFrame()
        for data in self.feature_data:
            for feature_name in self.feature_index_dict[data]:
                feature_index = self.feature_index_dict[data][feature_name]
                if self.feature_data[data] is not None:
                    if data == "jet":
                        for jet_index in range(self.max_jets):
                            if isinstance(feature_index, list):
                                for lep_index, index in enumerate(feature_index):
                                    feature_data[
                                        f"{feature_name}_{lep_index}_{jet_index}"
                                    ] = self.feature_data[data][:, jet_index, index]
                            else:
                                feature_data[f"{feature_name}_{jet_index}"] = (
                                    self.feature_data[data][:, jet_index, feature_index]
                                )
                    elif data == "lepton":
                        for lep_index in range(self.max_leptons):
                            feature_data[f"{feature_name}_{lep_index}"] = (
                                self.feature_data[data][:, lep_index, feature_index]
                            )
                    elif data == "global":
                        feature_data[f"{feature_name}"] = self.feature_data[data][
                            :, feature_index
                        ]
                    else:
                        continue
        feature_data = feature_data.replace(self.padding_value, np.nan)
        corr = feature_data.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr,
            annot=False,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax,
            center=0,
            cbar_kws={"ticks": np.linspace(-1, 1, 11)},
        )
        return fig, ax

    def enhance_feature_value(self, feature_name, enhancement_value):
        """
        Enhances the value of a specific feature in the feature data for samples with a label of 1.
        Parameters:
            feature_name (str): The name of the feature to enhance.
            enhancement_value (float): The value to set for the specified feature in samples with a label of 1.
        Raises:
            ValueError: If the feature data has not been prepared using the `prepare_data()` method.
            ValueError: If the specified feature name is not found in the feature index dictionary.
        Notes:
            - This method modifies the feature data in place.
            - This can be used for debbugging or testing purposes to artificially enhance the feature values for specific samples.
        """

        if self.feature_data is None:
            raise ValueError(
                "Feature data not prepared. Please prepare data using prepare_data() method."
            )
        if feature_name not in self.feature_index_dict:
            raise ValueError(
                f"Feature {feature_name} not found in feature index dictionary."
            )
        feature_index = self.feature_index_dict[feature_name]
        self.feature_data[:, :, feature_index] = np.where(
            self.labels == 1, enhancement_value, self.feature_data[:, :, feature_index]
        )

    def get_labels(self):
        if self.labels is None:
            raise ValueError(
                "Labels not prepared. Please prepare data using prepare_data() method."
            )
        return self.labels

    def get_feature_data(self, dataType, feature_name):
        """
        Retrieves feature data for a specified data type and feature name.
        Args:
            dataType (str): The type of data to retrieve. Possible values include
                            "jet", "lepton", "global", "non_training", and "event_weight".
            feature_name (str): The name of the feature to retrieve.
        Returns:
            numpy.ndarray: The requested feature data. The shape of the returned data
                           depends on the dataType:
                           - For "jet", "lepton", and "global": Returns a 3D array
                             with shape (samples, objects, feature_index).
                           - For "non_training": Returns a 2D array with shape
                             (samples, feature_index).
                           - For "event_weight": Returns the event weight data.
        Raises:
            ValueError: If the feature data is not prepared or if the specified
                        dataType is not found in the feature data.
        """

        if self.feature_data is None:
            raise ValueError(
                "Feature data not prepared. Please prepare data using prepare_data() method."
            )
        feature_index = self.feature_index_dict[dataType][feature_name]
        if dataType in self.feature_index_dict:
            if dataType == "jet":

                return self.feature_data[dataType][:, :, feature_index]
            elif dataType == "lepton":
                return self.feature_data[dataType][:, :, feature_index]
            elif dataType == "global":
                return self.feature_data[dataType][:, :, feature_index]
            elif dataType == "non_training":
                return self.feature_data[dataType][:, feature_index]
            elif dataType == "event_weight":
                return self.get_event_weight()

        else:
            raise ValueError(f"Data type {dataType} not found in feature data.")

    def normalise_data(self):
        """
        Normalises the feature data by subtracting the mean and dividing by the standard deviation
        for each feature type (lepton, jet, global) while ignoring padding values. The normalisation
        factors (mean and standard deviation) are stored for each feature type.
        Raises:
            ValueError: If feature data has not been prepared using the prepare_data() method.
        Notes:
            - The method skips normalisation for specific data types such as "non_training",
              "event_weight", "regression_targets", and "labels".
            - Padding values are excluded from the calculation of mean and standard deviation.
        Attributes:
            feature_data (dict): Dictionary containing feature data for different types.
            padding_value: Value used to identify padding in the data.
            data_normalisation_factors (dict): Dictionary to store normalisation factors
                                               (mean and standard deviation) for each feature type.
        """

        if self.feature_data is None:
            raise ValueError(
                "Feature data not prepared. Please prepare data using prepare_data() method."
            )
        for data in self.feature_data:
            if self.feature_data[data] is not None:
                if data == "lepton":
                    non_padding_mask = (
                        self.feature_data[data] != self.padding_value
                    ).all(axis=-1)
                    mean = np.mean(self.feature_data[data][non_padding_mask], axis=0)
                    std = np.std(self.feature_data[data][non_padding_mask], axis=0)
                    self.feature_data[data][non_padding_mask] = (
                        self.feature_data[data][non_padding_mask] - mean
                    ) / std
                    self.data_normalisation_factors[data] = {"mean": mean, "std": std}
                elif data == "jet":
                    non_padding_mask = (
                        self.feature_data[data] != self.padding_value
                    ).all(axis=-1)
                    mean = np.mean(self.feature_data[data][non_padding_mask], axis=0)
                    std = np.std(self.feature_data[data][non_padding_mask], axis=0)
                    self.feature_data[data][non_padding_mask] = (
                        self.feature_data[data][non_padding_mask] - mean
                    ) / std
                    self.data_normalisation_factors[data] = {"mean": mean, "std": std}
                elif data == "global":
                    non_padding_mask = (
                        self.feature_data[data] != self.padding_value
                    ).all(axis=-1)
                    mean = np.mean(self.feature_data[data][non_padding_mask], axis=0)
                    std = np.std(self.feature_data[data][non_padding_mask], axis=0)
                    self.feature_data[data][non_padding_mask] = (
                        self.feature_data[data][non_padding_mask] - mean
                    ) / std
                    self.data_normalisation_factors[data] = {"mean": mean, "std": std}
                elif data == "non_training":
                    continue
                elif data == "event_weight":
                    continue
                elif data == "regression_targets":
                    continue
                elif data == "labels":
                    continue

    def plot_feature_distribution(self, data, feature_name, file_name=None, **kwargs):
        """
        Plots the distribution of a specified feature from the prepared feature data.
        Parameters:
            data (str): The type of data to plot the feature distribution for.
                        Must be one of "jet", "lepton", or "global".
            feature_name (str): The name of the feature to plot.
            file_name (str, optional): The file path to save the plot. If None, the plot is not saved. Default is None.
            **kwargs: Additional keyword arguments to pass to `sns.histplot`.
        Raises:
            ValueError: If the feature data has not been prepared.
            ValueError: If the specified data type is not found in the feature index dictionary.
            ValueError: If the specified feature name is not found in the feature index dictionary.
            ValueError: If the specified data type is not found in the feature data.
        Returns:
            tuple: A tuple containing the matplotlib figure and axes objects (fig, ax).
        """

        if self.feature_data is None:
            raise ValueError(
                "Feature data not prepared. Please prepare data using prepare_data() method."
            )
        if data not in self.feature_index_dict:
            raise ValueError(f"Data type {data} not found in feature index dictionary.")
        if feature_name not in self.feature_index_dict[data]:
            raise ValueError(
                f"Feature {feature_name} not found in feature index dictionary."
            )
        feature_index = self.feature_index_dict[data][feature_name]
        if data == "jet":
            feature_data = self.feature_data[data][:, :, feature_index]
        elif data == "lepton":
            feature_data = self.feature_data[data][:, :, feature_index]
        elif data == "global":
            feature_data = self.feature_data[data][:, feature_index]
        else:
            raise ValueError(f"Data type {data} not found in feature data.")
        feature_data = feature_data[feature_data != self.padding_value]
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.histplot(feature_data, ax=ax, **kwargs)
        ax.set_title(f"Feature distribution for {feature_name}")
        if file_name is not None:
            plt.savefig(file_name)
        return fig, ax
