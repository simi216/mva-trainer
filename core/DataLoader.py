import uproot
import awkward as ak
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import convert_to_tensor

class DataLoader:
    def __init__(self, feature_clipping):
        self.features = list(feature_clipping.keys())
        self.clipping = feature_clipping
        self.data = None
        if not isinstance(self.clipping, dict):
            raise ValueError("Clipping should be a dictionary with feature names as keys and clipping values as values.")


    def load_data(self, file_path, tree_name, max_events=None):
        branches = self.features
        if not isinstance(file_path, str):
            raise ValueError("File path should be a string.")
        with uproot.open(file_path) as file:
            if tree_name not in file:
                raise ValueError(f"Tree {tree_name} not found in file {file_path}.")
            missing_branches = [branch for branch in branches if branch not in file[tree_name].keys()]
            if missing_branches:
                raise ValueError(f"The following branches are missing in the tree {tree_name}: {missing_branches}")
            if max_events is not None:
                data = file[tree_name].arrays(branches, library="ak", entry_stop=max_events)
            else:
                data = file[tree_name].arrays(branches, library="ak")
        if data is None:
            raise ValueError(f"No data found in {file_path} for tree {tree_name} with branches {branches}")
        self.data = data
        self._get_padded_data()


    def _get_padded_data(self, padding_value = -999):
        data_dict = {}
        for feature in self.features:
            if feature not in self.data.fields:
                raise ValueError(f"Feature {feature} not found in data.")
            if self.clipping[feature] == 1:
                data_dict[feature] = ak.fill_none(self.data[feature], padding_value)
                continue
            else:
                feature_variable = ak.fill_none(ak.pad_none(self.data[feature], self.clipping[feature], clip=True, axis = 1), padding_value)
                for i in range(self.clipping[feature]):
                    data_dict[f"{feature}_{i}"] = feature_variable[:, i]
        self.data = pd.DataFrame(data_dict)


    def get_data(self):
        if self.data is None:
            raise ValueError("Data not loaded. Please load data using load_data() method.")
        return self.data


class DataPreprocessor():
    def __init__(self, jet_features : list[str], lepton_features : list[str], jet_truth_label : str, lepton_truth_label : str, max_leptons : int =2, max_jets : int =4, global_features : list[str]=None, non_training_features : list[str]=None, regression_targets : list[str] = None,  event_weight : str = None):
        self.jet_features = jet_features
        self.lepton_features = lepton_features
        self.jet_truth_label = jet_truth_label
        self.lepton_truth_label = lepton_truth_label
        self.max_leptons = max_leptons
        self.max_jets = max_jets
        self.global_features = global_features
        self.n_jets : int= len(jet_features)
        self.n_leptons : int  = len(lepton_features)
        self.n_global : int = len(global_features) if global_features else 0
        self.data = None
        self.data_length = None
        self.padding_value = -999.0
        self.combined_features = None
        self.n_combined : int = 0
        self.comb_feature_jet_vars = None
        self.comb_feature_lep_vars = None
        self.comb_feature_computation = None
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

    def apply_cut(self, cut_feature, cut_low = None, cut_high = None):
        if cut_low is None and cut_high is None:
            raise ValueError("At least one of cut_low or cut_high must be specified.")
        if cut_low is not None and cut_high is not None and cut_low >= cut_high:
            raise ValueError("cut_low must be less than cut_high.")
        self.cut_dict[cut_feature] = (cut_low, cut_high)

    def compute_pairwise_features(self):
        if self.data is None:
            raise ValueError("Data not loaded. Please load data using load_data() method.")
        for feature in self.combined_features:
            for lep_index in range(self.max_leptons):
                for jet_index in range(self.max_jets):
                    padding_mask = (self.data[[jet_var + f"_{jet_index}" for jet_var in self.comb_feature_jet_vars[feature]] + [lepton_var + f"_{lep_index}" for lepton_var in self.comb_feature_lep_vars[feature]]] != self.padding_value).all(axis=1)
                    self.data.loc[padding_mask, feature + f"_{lep_index}_{jet_index}"]= self.comb_feature_computation[feature](*(self.data.loc[padding_mask,jet_var + f"_{jet_index}"] for jet_var in self.comb_feature_jet_vars[feature]), *(self.data.loc[padding_mask,lepton_var + f"_{lep_index}"] for lepton_var in self.comb_feature_lep_vars[feature]))
                    self.data.loc[~padding_mask, feature + f"_{lep_index}_{jet_index}"] = self.padding_value


    def reorder_by_feature(self, reorder_feature):
        if self.feature_data is None:
            raise ValueError("Feature data not prepared. Please prepare data using prepare_data() method.")
        if reorder_feature not in self.feature_index_dict:
            raise ValueError(f"Feature {reorder_feature} not found in feature index dictionary.")
        if self.labels is None:
            raise ValueError("Labels not prepared. Please prepare data using prepare_data() method.")
        feature_index = self.feature_index_dict[reorder_feature]
        for event_index in range(self.feature_data.shape[0]):
            # Extract the feature values for the jets in the current event
            feature_values = self.feature_data[event_index, :, feature_index]
            # Identify unmasked jets (those not equal to the padding value)
            unmasked_indices = np.where(feature_values != self.padding_value)[0]
            # Get the sorting indices for unmasked jets based on the feature values
            sorted_indices = unmasked_indices[np.argsort(feature_values[unmasked_indices])]
            # Create a new ordering array that keeps masked jets in place
            full_sorted_indices = np.arange(self.feature_data.shape[1])
            full_sorted_indices[unmasked_indices] = sorted_indices

            # Reorder the feature data and labels based on the sorted indices
            self.feature_data[event_index] = self.feature_data[event_index, full_sorted_indices]
            self.labels[event_index] = self.labels[event_index, full_sorted_indices]


    def register_pairwise_features(self, combined_features : list[str], jet_features : dict[str, list[str]], lepton_features : dict[str, list[str]] , feature_computation : dict[str, callable]):
        if self.data is not None:
            raise ValueError("Data already loaded. Please use a different instance of the class to register new features.")
        for feature in combined_features:
            if feature not in jet_features:
                raise ValueError(f"Feature {feature} not found in jet_features.")
            if feature not in lepton_features:
                raise ValueError(f"Feature {feature} not found in lepton_features.")
            if feature not in feature_computation:
                raise ValueError(f"Feature {feature} not found in feature_computation.")
        self.comb_feature_jet_vars = jet_features
        self.comb_feature_lep_vars = lepton_features
        self.comb_feature_computation = feature_computation
        self.combined_features = combined_features
        self.n_combined = len(combined_features)
        needed_jet_features = set()
        needed_lepton_features = set()
        for feature in combined_features:
            needed_jet_features.update(self.comb_feature_jet_vars[feature])
            needed_lepton_features.update(self.comb_feature_lep_vars[feature])
        self.load_jet_features = list(set(self.jet_features) | needed_jet_features)
        self.load_lepton_features = list(set(self.lepton_features) | needed_lepton_features)


    def load_data(self, file_path, tree_name, max_events=None, cut_neg_weights=True):
        if self.data is None:
            feature_clipping = {feature : self.max_jets for feature in self.load_jet_features}
            feature_clipping.update({feature : self.max_leptons for feature in self.load_lepton_features})
            feature_clipping.update({feature : 1 for feature in self.global_features}) if self.global_features else None
            feature_clipping.update({self.jet_truth_label : 6})
            feature_clipping.update({self.lepton_truth_label : 2})
            feature_clipping.update({feature :  1 for feature in self.non_training_features}) if self.non_training_features else None
            feature_clipping.update({self.event_weight_label : 1}) if self.event_weight_label else None
            feature_clipping.update({regression_target : 1 for regression_target in self.regression_targets}) if self.regression_targets else None
            DataHandle = DataLoader(feature_clipping)
            DataHandle.load_data(file_path, tree_name, max_events=max_events)
            self.data = DataHandle.get_data()
            DataHandle = None  # Clear the DataLoader instance to save memory
            if self.cut_dict:
                for cut_feature, (cut_low, cut_high) in self.cut_dict.items():
                    if cut_feature not in self.data.columns:
                        raise ValueError(f"Cut feature {cut_feature} not found in data.")
                    print(f"Applying cut on feature {cut_feature}: low={cut_low}, high={cut_high}")
                    print(f"{self.data[cut_feature].min()} <= {cut_feature} <= {self.data[cut_feature].max()}")
                    if cut_low is not None:
                        self.data = self.data[self.data[cut_feature] >= cut_low]
                    if cut_high is not None:
                        self.data = self.data[self.data[cut_feature] <= cut_high]
            if self.event_weight_label is not None and cut_neg_weights:
                self.data = self.data[self.data[self.event_weight_label] >= 0]
            self.data_length = len(self.data)
        else:
            raise ValueError("Data already loaded. Please use a different instance of the class to load new data.")


    def prepare_data(self):
        if self.data is None:
            raise ValueError("Data not loaded. Please load data using load_data() method.")
        if self.combined_features is not None:
            self.compute_pairwise_features()
            print("Pairwise features computed.")
        self.build_pairs()
        #self.reorder_by_feature("dR_lep_jet")


    def build_pairs(self):
        if self.data is None:
            raise ValueError("Data not loaded. Please load data using load_data() method.")

        lepton_data = self.data[[lepton_var + f"_{lep_index}" for lepton_var in self.lepton_features for lep_index in range(self.max_leptons)]].to_numpy().reshape(self.data_length, -1, self.max_leptons).transpose((0, 2, 1))
        lepton_indices = {lepton_var: idx for idx, lepton_var in enumerate(self.lepton_features)}

        if self.combined_features is not None:
            jet_data = self.data[[jet_var + f"_{jet_index}" for jet_var in self.jet_features for jet_index in range(self.max_jets)] +
                        [comb_var + f"_{lep_index}_{jet_index}" for comb_var in self.combined_features for lep_index in range(self.max_leptons) for jet_index in range(self.max_jets)]].to_numpy().reshape(self.data_length, -1, self.max_jets).transpose((0, 2, 1))
            jet_indices = {jet_var: idx for idx, jet_var in enumerate(self.jet_features)}
            combined_indices = {comb_var: [self.n_jets + self.max_leptons * idx + i   for i in range(self.max_leptons)] for idx, comb_var in enumerate(self.combined_features)}
            jet_indices.update(combined_indices)
        else:
            jet_data = self.data[[jet_var + f"_{jet_index}" for jet_var in self.jet_features for jet_index in range(self.max_jets)]].to_numpy().reshape(self.data_length, -1, self.max_jets).transpose((0, 2, 1))
            jet_indices = {jet_var: idx for idx, jet_var in enumerate(self.jet_features)}

        global_data = self.data[[global_var for global_var in self.global_features]].to_numpy().reshape(self.data_length, 1, self.n_global) if self.global_features else None
        global_indices = {global_var: idx for idx, global_var in enumerate(self.global_features)} if self.global_features else {}
        non_training_data = self.data[[non_training_var for non_training_var in self.non_training_features]].to_numpy() if self.non_training_features else None
        non_training_indices = {non_training_var: idx for idx, non_training_var in enumerate(self.non_training_features)} if self.non_training_features else {}

        feature_dict = {}

        self.feature_index_dict.update({"lepton" : lepton_indices})
        feature_dict.update({"lepton": lepton_data})
        self.feature_index_dict.update({"jet" : jet_indices})
        feature_dict.update({"jet": jet_data})
        if self.global_features is not None:
            self.feature_index_dict.update({"global" : global_indices})
            feature_dict.update({"global": global_data})
        if self.non_training_features is not None:
            self.feature_index_dict.update({"non_training" : non_training_indices})
            feature_dict.update({"non_training": non_training_data})
        if self.event_weight_label is not None:
            feature_dict.update({"event_weight": self.data[self.event_weight_label].to_numpy()})
            self.feature_index_dict.update({"event_weight" : {self.event_weight_label: 0}})
        if self.regression_targets is not None:
            feature_dict.update({"regression_targets": self.data[self.regression_targets].to_numpy()})
            self.feature_index_dict.update({"regression_targets" : {regression_target: idx for idx, regression_target in enumerate(self.regression_targets)}})
        self.feature_data = feature_dict

        self.build_labels()
        self.data = None  # Clear the original data to save memory


    def get_event_weight(self, cut_neg_weights=False):
        if self.feature_data is None:
            raise ValueError("Feature Data not loaded. Please load data using load_data() method.")
        if self.event_weight_label is None:
            raise ValueError("Event weight label not provided.")
        event_weight = self.feature_data["event_weight"]
        if cut_neg_weights:
            event_weight = event_weight[event_weight >= 0]
        event_weight = event_weight / np.sum(event_weight)
        self.data_length = len(event_weight)

        return event_weight


    def build_labels(self):
        if self.data is None:
            raise ValueError("Data not loaded. Please load data using load_data() method.")

        jet_truth = self.data[[self.jet_truth_label +f"_{0}", self.jet_truth_label +f"_{3}"]].to_numpy()
        lepton_truth = self.data[[self.lepton_truth_label +f"_{0}", self.lepton_truth_label +f"_{1}"]].to_numpy()

        reco_success_mask = (jet_truth != -1).all(axis=1) & (jet_truth < self.max_jets).all(axis=1)

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
                    if jet_truth[event_index, 0] == jet_index and lepton_truth[event_index, 0] == lep_index:
                        pair_truth[event_index, jet_index, lep_index] = 1
                    elif jet_truth[event_index, 1] == jet_index and lepton_truth[event_index, 1] == lep_index:
                        pair_truth[event_index, jet_index, lep_index] = 1
                    else:
                        pair_truth[event_index, jet_index, lep_index] = 0

        if self.regression_targets is not None:
            self.labels = self.feature_data["regression_targets"]
        else:
            self.labels = pair_truth
        self.feature_data["labels"] = pair_truth


    def split_data(self, test_size=0.2, random_state=42):
        if self.feature_data is None:
            raise ValueError("Feature data not prepared. Please prepare data using prepare_data() method.")
        if self.labels is None:
            raise ValueError("Labels not prepared. Please prepare data using prepare_data() method.")
        from sklearn.model_selection import train_test_split
        X_train = {}
        X_test = {}
        y_train = None
        y_test = None
        for data in self.feature_data:
            if self.feature_data[data] is not None:
                X_train[data], X_test[data], y_train, y_test = train_test_split(self.feature_data[data], self.labels, test_size=test_size, random_state=random_state)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


    def get_data(self):
        if self.X_train is None or self.X_test is None:
            self.split_data()
        return self.X_train, self.y_train, self.X_test, self.y_test


    def create_k_folds(self, n_folds: int = 5, n_splits: int = 1, random_state: int = 42) -> list[tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray]]:
        '''
        Create k-folds using views (not copies), assuming data is shuffled and splits are contiguous.
        
        Args:
            n_folds (int): Number of folds per split.
            n_splits (int): Number of random splits of the data.
            random_state (int): Seed for reproducibility.
        
        Returns:
            folded_data (list): List of (X_train, y_train, X_test, y_test) using views.
        '''
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
            end = (split_index + 1) * split_size if split_index < n_splits - 1 else self.data_length

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
                    k: np.concatenate([
                        self.feature_data[k][train1_start:train1_end],
                        self.feature_data[k][train2_start:train2_end]
                    ]) for k in self.feature_data
                }
                y_train = np.concatenate([
                    self.labels[train1_start:train1_end],
                    self.labels[train2_start:train2_end]
                ])
                X_test = {
                    k: self.feature_data[k][test_start:test_end]
                    for k in self.feature_data
                }
                y_test = self.labels[test_start:test_end]

                folded_data.append((X_train, y_train, X_test, y_test))

        return folded_data


    def plot_feature_correlation(self):
        if self.feature_data is None:
            raise ValueError("Feature data not prepared. Please prepare data using prepare_data() method.")
        feature_data = pd.DataFrame()
        for data in self.feature_data:
            for feature_name in self.feature_index_dict[data]:
                feature_index = self.feature_index_dict[data][feature_name]
                if self.feature_data[data] is not None:
                    if data == "jet":
                        for jet_index in range(self.max_jets):
                            if isinstance(feature_index, list):
                                for lep_index, index in enumerate(feature_index):
                                    feature_data[f"{feature_name}_{lep_index}_{jet_index}"] = self.feature_data[data][:, jet_index, index]
                            else:
                                feature_data[f"{feature_name}_{jet_index}"] = self.feature_data[data][:, jet_index, feature_index]
                    elif data == "lepton":
                        for lep_index in range(self.max_leptons):
                            feature_data[f"{feature_name}_{lep_index}"] = self.feature_data[data][:, lep_index, feature_index]
                    elif data == "global":
                        feature_data[f"{feature_name}"] = self.feature_data[data][:, feature_index]
                    else: continue
        feature_data = feature_data.replace(self.padding_value, np.nan)
        corr = feature_data.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=False, fmt=".2f", cmap="coolwarm", ax=ax, center=0, cbar_kws={"ticks": np.linspace(-1, 1, 11)})
        return fig, ax


    def enhance_feature_value(self, feature_name, enhancement_value):
        if self.feature_data is None:
            raise ValueError("Feature data not prepared. Please prepare data using prepare_data() method.")
        if feature_name not in self.feature_index_dict:
            raise ValueError(f"Feature {feature_name} not found in feature index dictionary.")
        feature_index = self.feature_index_dict[feature_name]
        self.feature_data[:, :, feature_index] = np.where(self.labels == 1, enhancement_value, self.feature_data[:, :, feature_index])


    def get_labels(self):
        if self.labels is None:
            raise ValueError("Labels not prepared. Please prepare data using prepare_data() method.")
        return self.labels


    def get_feature_data(self, dataType , feature_name):
        if self.feature_data is None:
            raise ValueError("Feature data not prepared. Please prepare data using prepare_data() method.")
        feature_index = self.feature_index_dict[dataType][feature_name]
        if dataType in self.feature_index_dict:
            if dataType == "jet":
                
                return self.feature_data[dataType][:, :, feature_index]
            elif dataType == "lepton":
                return self.feature_data[dataType][:,:, feature_index]
            elif dataType == "global":
                return self.feature_data[dataType][:,:, feature_index]
            elif dataType == "non_training":
                return self.feature_data[dataType][:, feature_index]
            elif dataType == "event_weight":
                return self.get_event_weight()

        else:
            raise ValueError(f"Data type {dataType} not found in feature data.")


    def normalise_data(self):
        if self.feature_data is None:
            raise ValueError("Feature data not prepared. Please prepare data using prepare_data() method.")
        for data in self.feature_data:
            if self.feature_data[data] is not None:
                if data == "lepton":
                    non_padding_mask = (self.feature_data[data] != self.padding_value).all(axis=-1)
                    mean = np.mean(self.feature_data[data][non_padding_mask], axis=0)
                    std = np.std(self.feature_data[data][non_padding_mask], axis=0)
                    self.feature_data[data][non_padding_mask] = (self.feature_data[data][non_padding_mask] - mean) / std
                    self.data_normalisation_factors[data] = {"mean": mean, "std": std}
                elif data == "jet":
                    non_padding_mask = (self.feature_data[data] != self.padding_value).all(axis=-1)
                    mean = np.mean(self.feature_data[data][non_padding_mask], axis=0)
                    std = np.std(self.feature_data[data][non_padding_mask], axis=0)
                    self.feature_data[data][non_padding_mask] = (self.feature_data[data][non_padding_mask] - mean) / std
                    self.data_normalisation_factors[data] = {"mean": mean, "std": std}
                elif data == "global":
                    non_padding_mask = (self.feature_data[data] != self.padding_value).all(axis=-1)
                    mean = np.mean(self.feature_data[data][non_padding_mask], axis=0)
                    std = np.std(self.feature_data[data][non_padding_mask], axis=0)
                    self.feature_data[data][non_padding_mask] = (self.feature_data[data][non_padding_mask] - mean) / std
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
        if self.feature_data is None:
            raise ValueError("Feature data not prepared. Please prepare data using prepare_data() method.")
        if data not in self.feature_index_dict:
            raise ValueError(f"Data type {data} not found in feature index dictionary.")
        if feature_name not in self.feature_index_dict[data]:
            raise ValueError(f"Feature {feature_name} not found in feature index dictionary.")
        feature_index = self.feature_index_dict[data][feature_name]
        if data == "jet":
            feature_data = self.feature_data[data][:, :, feature_index]
        elif data == "lepton":
            feature_data = self.feature_data[data][:,:, feature_index]
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