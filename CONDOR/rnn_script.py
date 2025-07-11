import sys
sys.path.append("../")
import os

import tensorflow as tf
import keras
import numpy as np
from importlib import reload
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


import core.AssignmentRNN as AssignmentRNN
from core.DataLoader import DataLoader, DataPreprocessor
from core.CustomObjects import TemporalSoftmax
import core.AssignmentKFold

MAX_JETS = 4
DIR_NAME = "plots_rnn_hpo/"

def main(argv):
    if len(argv) < 5:
        print("Usage: python rnn_script_1.py lstm_size jet_embedding lep_embedding")
        sys.exit(1)
    lstm_size = int(argv[1])
    jet_embedding = int(argv[2])
    lep_embedding = 4
    lambda_excl = float(argv[3])
    droupout_rate = float(argv[4]) if len(argv) > 4 else 0.05



    PLOTS_DIR = f"{DIR_NAME}/{lstm_size}_{jet_embedding}_{lambda_excl}/"
    if not os.path.exists(DIR_NAME):
        os.makedirs(DIR_NAME)
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)


    DataProcessor = DataPreprocessor(jet_features=["ordered_jet_pt", "ordered_jet_e", "ordered_jet_phi", "ordered_jet_eta", "ordered_jet_b_tag","m_l1j", "m_l2j", "dR_l1j", "dR_l2j"], 
                                    lepton_features=["lep_pt","lep_e", "lep_eta", "lep_phi"],
                                    jet_truth_label="ordered_event_jet_truth_idx", 
                                    lepton_truth_label="event_lepton_truth_idx", 
                                    global_features = ["met_met_NOSYS","met_phi_NOSYS"], 
                                    max_leptons=2, 
                                    max_jets = MAX_JETS, 
                                    non_training_features= ["truth_ttbar_mass", "truth_ttbar_pt", "N_jets"], 
                                    event_weight="weight_mc_NOSYS")

    DataProcessor.load_data("/data/dust/group/atlas/ttreco/full_training.root", "reco", max_events = 4000000)
    DataProcessor.prepare_data()
    DataProcessor.normalise_data()


    JetMatcher = core.AssignmentKFold.KFoldEvaluation(AssignmentRNN.RNNJetMatcher,DataProcessor,random_state = 42, n_splits = 5, n_folds = 4)

    JetMatcher.build_models(
        lstm_size = lstm_size,
        jet_embedding_dim = jet_embedding,
        jet_layers = 3,
        lepton_embedding_dim = lep_embedding,
        lepton_layers = 3,
        cross_attention_dim = 8,
        cross_layers = 3,
        n_heads = 4,
        lambda_excl = lambda_excl,
        dropout_rate=droupout_rate
    )

    JetMatcher.compile_models(lambda_excl = lambda_excl, optimizer = keras.optimizers.Adam(learning_rate=0.001))

    JetMatcher.train_models(epochs=50,
                            batch_size=512,
                            verbose=0,
                            weight = "sample")


    fig_list, ax_list = JetMatcher.plot_history()
    for i in range(len(fig_list)):
        fig_list[i].savefig(PLOTS_DIR + f"training_history_{i}.pdf")
        fig_list[i].clear()
        plt.close(fig_list[i])

    #JetMatcher.save_model(PLOTS_DIR + "jet_matcher.keras")

    fig, ax = JetMatcher.plot_confusion_matrix()
    fig.suptitle("Transformer-based matching")
    fig.savefig(PLOTS_DIR + "TransformerJetMatcherConfusionMatrix.pdf")

    #JetMatcher.plot_permutation_importance()[0].savefig(PLOTS_DIR + "Transformer_permutation_importance.pdf")

    '''
    DataProcessor = DataPreprocessor(["ordered_jet_pt", "ordered_jet_e", "ordered_jet_eta", "ordered_jet_phi", "ordered_jet_b_tag", "m_l1j", "m_l2j", "dR_l1j", "dR_l2j"], ["lep_pt","lep_e","lep_eta","lep_phi", "lep_charge"],"ordered_event_jet_truth_idx", "event_lepton_truth_idx", global_features = ["met_met_NOSYS","met_phi_NOSYS"], non_training_features=["truth_ttbar_mass", "truth_ttbar_pt", "N_jets", "weight_mc_NOSYS"], max_leptons=2, max_jets = MAX_JETS)

    DataProcessor.load_data("/data/dust/group/atlas/ttreco/full_training.root", "reco", max_events = 20000)
    DataProcessor.prepare_data()

    jet_b_tag = DataProcessor.get_feature_data("jet", "ordered_jet_b_tag")
    jet_pt = DataProcessor.get_feature_data("jet", "ordered_jet_pt")
    jet_dR_lep_jet = np.stack([DataProcessor.get_feature_data("jet", "dR_l1j"), DataProcessor.get_feature_data("jet", "dR_l2j")],axis = -1)
    #jet_dR = DataProcessor.get_feature_data("jet", "dR_lep_jet")
    lep_charge = DataProcessor.get_feature_data("lepton", "lep_charge")
    lep_pt = DataProcessor.get_feature_data("lepton", "lep_pt")
    lep_e = DataProcessor.get_feature_data("lepton", "lep_e")
    lep_eta = DataProcessor.get_feature_data("lepton", "lep_eta")
    lep_phi = DataProcessor.get_feature_data("lepton", "lep_phi")
    ttbar_mass = DataProcessor.get_feature_data("non_training", "truth_ttbar_mass")
    ttbar_pt = DataProcessor.get_feature_data("non_training", "truth_ttbar_pt")
    N_jets = DataProcessor.get_feature_data("non_training", "N_jets")
    mc_weight = DataProcessor.get_feature_data("non_training", "weight_mc_NOSYS")
    mL_lep_jet = np.stack([DataProcessor.get_feature_data("jet", "m_l1j"), DataProcessor.get_feature_data("jet", "m_l2j")],axis = -1)
    #mL_lep_jet = DataProcessor.get_feature_data("jet", "m_lep_jet")
    truth_idx = DataProcessor.get_labels()
    b_tag_indices = np.empty((jet_b_tag.shape[0], 2), dtype = int)

    def dR_jet_matcher(jet_b_tag, jet_dR, jet_pt):
        jet_matcher_indices = np.empty((jet_b_tag.shape[0], 2), dtype = int)
        for i in range(len(jet_b_tag)):
            valid_indices = np.where(jet_b_tag[i] != -999)[0]
            b_tag_indices = []
            if len(valid_indices) < 2:
                continue
            if np.sum(jet_b_tag[i, valid_indices] > 2) >= 2:
                b_tag_indices = valid_indices[np.where(jet_b_tag[i, valid_indices] > 2)[0]]
            elif np.sum(jet_b_tag[i, valid_indices] > 2) == 1:
                b_tag_indices.append(valid_indices[np.argsort(jet_b_tag[i, valid_indices])[-1]])
                b_tag_indices.append(valid_indices[np.where(jet_b_tag[i, valid_indices] <= 2)[0][0]])
            else:
                b_tag_indices.append(valid_indices[np.argsort(jet_pt[i, valid_indices])[-1]])
                b_tag_indices.append(valid_indices[np.argsort(jet_pt[i, valid_indices])[-2]])
            # Find the (jet, lepton) pair with the smallest dR
            dR_sub = jet_dR[i, valid_indices, :]
            jet_idx, lep_idx = np.unravel_index(np.argmin(dR_sub), dR_sub.shape)
            first_jet = valid_indices[jet_idx]
            jet_matcher_indices[i][lep_idx] = first_jet

            # Remove the assigned jet from valid_indices and assign the remaining lepton to the next best jet
            remaining_lep = 1 - lep_idx
            remaining_jets = np.delete(valid_indices, jet_idx)
            next_jet_idx = remaining_jets[np.argmin(jet_dR[i, remaining_jets, remaining_lep])]
            jet_matcher_indices[i][remaining_lep] = next_jet_idx
        return jet_matcher_indices

    truth_index = np.argmax(truth_idx, axis = 1)
    dR_matched_indices = dR_jet_matcher(jet_b_tag, jet_dR_lep_jet, jet_pt)
    ml_matched_indices = dR_jet_matcher(jet_b_tag, mL_lep_jet, jet_pt)


    fig, ax = JetMatcher.accuracy_vs_feature("truth_ttbar_mass", xlims = (350e3, 800e3), bins = 20)
    BaseModel.plot_accuracy_feature(ml_matched_indices, truth_index, ttbar_mass, r"$m_{tt}$ [GeV]", xlims = (350e3, 800e3), fig = fig, ax = ax, accuracy_color ="tab:red", label = "mL-matching", event_weights = mc_weight/np.sum(mc_weight))
    BaseModel.plot_accuracy_feature(dR_matched_indices, truth_index, ttbar_mass, r"$m_{tt}$ [GeV]", xlims = (350e3, 800e3), fig = fig, ax = ax, accuracy_color ="tab:green", label = "dR-matching", event_weights = mc_weight/np.sum(mc_weight))
    ax.legend()
    fig.savefig(PLOTS_DIR + "accuracy_vs_truth_ttbar_mass.pdf")


    fig, ax = JetMatcher.accuracy_vs_feature("N_jets", xlims = (2,5), bins = 3)
    BaseModel.plot_accuracy_feature(ml_matched_indices, truth_index, N_jets, r"#jets", xlims = (2,5), bins = 3, fig = fig, ax = ax, accuracy_color ="tab:red", label = "mL-matching", event_weights = mc_weight/np.sum(mc_weight))
    BaseModel.plot_accuracy_feature(dR_matched_indices, truth_index, N_jets, r"#jets", xlims = (2,5), bins = 3, fig = fig, ax = ax, accuracy_color ="tab:green", label = "dR-matching", event_weights = mc_weight/np.sum(mc_weight))
    ax.legend()
    fig.savefig(PLOTS_DIR + "accuracy_vs_N_jets.pdf")

    fig, ax = JetMatcher.accuracy_vs_feature("truth_ttbar_pt", xlims = (0, 300e3), bins = 20)
    BaseModel.plot_accuracy_feature(ml_matched_indices, truth_index, ttbar_pt, r"$p_{T}(tt)$ [GeV]", xlims = (0, 300e3), fig = fig, ax = ax, accuracy_color ="tab:red", label = "mL-matching", event_weights = mc_weight/np.sum(mc_weight))
    BaseModel.plot_accuracy_feature(dR_matched_indices, truth_index, ttbar_pt, r"$p_{T}(tt)$ [GeV]", xlims = (0, 300e3), fig = fig, ax = ax, accuracy_color ="tab:green", label = "dR-matching", event_weights = mc_weight/np.sum(mc_weight))
    ax.legend()
    fig.savefig(PLOTS_DIR + "accuracy_vs_truth_ttbar_pt.pdf")
    '''

    lep_1_accuracy, lep_1_accuracy_err, lep_2_accuracy, lep_2_accuracy_err, combined_accuracy, combined_accuracy_err = JetMatcher.evaluate_accuracy()
    with open(PLOTS_DIR + "accuracies.dat", "w") as f:
        f.write(f"{lep_1_accuracy} {lep_1_accuracy_err} {lep_2_accuracy} {lep_2_accuracy_err} {combined_accuracy} {combined_accuracy_err}\n")

    duplicate_jets, duplicate_jets_err = JetMatcher.duplicate_jets()
    with open(PLOTS_DIR + "duplicate_jets.dat", "w") as f:
        f.write(f"{duplicate_jets},{duplicate_jets_err}\n")

    JetMatcher.save_model(PLOTS_DIR + "jet_matcher.keras")


if __name__ == "__main__":
    main(sys.argv)
