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
ROOT_DIR = f"/afs/desy.de/user/a/aulich/mva-trainer/"
PLOTS_DIR = ROOT_DIR + "plots/"
MODEL_DIR = ROOT_DIR + "models/"
MODEL_NAME = "Raw_Transformer_Assignment"


import os
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)
if not os.path.exists(PLOTS_DIR + MODEL_NAME + "/"):
    os.makedirs(PLOTS_DIR + MODEL_NAME + "/")
PLOTS_DIR = PLOTS_DIR + MODEL_NAME + "/"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
if not os.path.exists(MODEL_DIR + MODEL_NAME + "/"):
    os.makedirs(MODEL_DIR + MODEL_NAME + "/")
MODEL_DIR = MODEL_DIR + MODEL_NAME + "/"


config = DataConfig(jet_features=["ordered_jet_pt","ordered_jet_eta", "ordered_jet_phi","ordered_jet_e","ordered_jet_b_tag"], 
                                lepton_features=["lep_pt", "lep_eta", "lep_phi", "lep_e"],
                                jet_truth_label="ordered_event_jet_truth_idx", 
                                lepton_truth_label="event_lepton_truth_idx", 
                                global_features = ["met_met_NOSYS","met_phi_NOSYS"], 
                                max_leptons=2, 
                                max_jets = MAX_JETS, 
                                non_training_features =["truth_ttbar_mass", "truth_ttbar_pt", "N_jets"], 
                                event_weight="weight_mc_NOSYS")

DataProcessor = DataPreprocessor(config)
DataProcessor.load_data("/data/dust/group/atlas/ttreco/full_training.root", "reco", max_events=4000000)
#DataProcessor.normalise_data()
X_train,y_train, X_val, y_val = DataProcessor.split_data(test_size=0.1, random_state=42)

TransformerMatcher = Models.FeatureConcatTransformer(config, name="Transformer")


TransformerMatcher.build_model(
    num_heads=8,
    hidden_dim=64,
    num_layers=7,
    dropout_rate=0.1
)
TransformerMatcher.adapt_normalization_layers(X_train)

TransformerMatcher.compile_model(
    loss = core.utils.AssignmentLoss(lambda_excl=0), optimizer=keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4), metrics=[core.utils.AssignmentAccuracy()]
)


TransformerMatcher.train_model(epochs=50,
                                X_train=X_train,
                                y_train=y_train,
                                sample_weights=core.utils.compute_sample_weights(X_train, y_train),
                                batch_size=128,
                                callbacks = keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True, mode ="min"))

TransformerMatcher.save_model("Transformer_Assignment.keras")
TransformerMatcher.export_to_onnx("Transformer_Assignment.onnx")

pred_val = TransformerMatcher.predict_indices(X_val)


from sklearn.metrics import ConfusionMatrixDisplay
fig, ax = plt.subplots(figsize=(8,8))
ConfusionMatrixDisplay.from_predictions(y_val[:,:,1].argmax(axis=-1),pred_val[:,:,1].argmax(axis=-1), normalize="true", ax=ax)
plt.savefig(PLOTS_DIR+"/confusion_matrix_lepton.png")