import sys
sys.path.append("..") # Ensure the parent directory is in the path

import core.assingment as Models
from core.DataLoader import DataPreprocessor, DataConfig
import core
import numpy as np
from importlib import reload
import keras
import matplotlib.pyplot as plt
MAX_JETS = 6
ROOT_DIR = f"/afs/desy.de/user/a/aulich/mva_trainer/"
PLOTS_DIR = ROOT_DIR + "plots/"

import os
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)
if not os.path.exists(PLOTS_DIR + "RNN/"):
    os.makedirs(PLOTS_DIR + "RNN/")
MODEL_DIR = ROOT_DIR + "models/"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(MODEL_DIR + "RNN/"):
    os.makedirs(MODEL_DIR + "RNN/")

config = DataConfig(jet_features=["ordered_jet_pt", "ordered_jet_e", "ordered_jet_phi", "ordered_jet_eta", "ordered_jet_b_tag", "dR_l1j", "dR_l2j"], 
                                lepton_features=["lep_pt","lep_e", "lep_eta", "lep_phi"],
                                jet_truth_label="ordered_event_jet_truth_idx", 
                                lepton_truth_label="event_lepton_truth_idx", 
                                global_features = ["met_met_NOSYS","met_phi_NOSYS"], 
                                max_leptons=2, 
                                max_jets = MAX_JETS, 
                                non_training_features =["truth_ttbar_mass", "truth_ttbar_pt", "N_jets"], 
                                event_weight="weight_mc_NOSYS")

DataProcessor = DataPreprocessor(config)
DataProcessor.load_data("/data/dust/group/atlas/ttreco/full_training.root", "reco", max_events=1000000)
DataProcessor.normalise_data()
X_train,y_train, X_val, y_val = DataProcessor.split_data(test_size=0.1, random_state=42)

reload(Models)
reload(core)
TransformerMatcher = Models.FeatureConcatRNN(config, name="RNN")

TransformerMatcher.build_model(
    hidden_dim=32,
    num_layers=4,
    dropout_rate=0.1
)
TransformerMatcher.compile_model(
    loss = core.utils.AssignmentLoss(lambda_excl=0), optimizer=keras.optimizers.Adam(learning_rate=1e-4), metrics=[core.utils.AssignmentAccuracy()]
)
TransformerMatcher.model.summary()
TransformerMatcher.load_model("RNN_Assignment.keras")

TransformerMatcher.train_model(epochs=100,
                                X_train=X_train,
                                y_train=y_train,
                                batch_size=512,
                                callbacks = keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True, mode ="max"))


TransformerMatcher.save_model("RNN_Assignment.keras")