#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <functional>
#include <memory>
#include <cmath>
#include "reco_mc_20.h"
#include <filesystem>

#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>
#include <TChain.h>
#include <TSystem.h>
#include <TString.h>
#include <TBranch.h>
#include <TLeaf.h>
#include <TObject.h>
#include <TLorentzVector.h>

typedef reco EventType;

class PreProcessor {
    public:
        PreProcessor(const char* inputFileName, const char* outputFileName, const char* treeName = "reco");
        ~PreProcessor();
        void Process();
        void RegisterNuFlowResults();
        void RegisterInitialStateInfo();


    private:
        void RegisterBranches();
        bool PreSelection();
        void FillBranches();

        void GetNuFlowBranches();
        void SaveNuFlowTTbarMass(TLorentzVector& l1, TLorentzVector& l2, TLorentzVector& j1, TLorentzVector& j2);
        void FillInitialStateBranches();

        TFile* inputFile;
        TFile* outputFile;
        TChain* inputTree;
        TTree* outputTree;
        std::string TreeName;
        std::string inputFileName;
        std::string outputFileName;


        // Branches
        std::vector<double> lep_pt;
        std::vector<double> lep_eta;
        std::vector<double> lep_phi;
        std::vector<double> lep_e;
        std::vector<double> lep_charge;
        std::vector<double> lep_pid;
        std::vector<int> event_lepton_truth_idx;

        std::vector<double> jet_pt;
        std::vector<double> jet_eta;
        std::vector<double> jet_phi;
        std::vector<double> jet_e;
        std::vector<double> jet_btag;
        std::vector<int> event_jet_truth_idx;

        std::vector<double> m_l1j;
        std::vector<double> m_l2j;

        std::vector<double> dR_l1j;
        std::vector<double> dR_l2j;

        double truth_ttbar_mass;
        double truth_ttbar_pt;

        double truth_top_mass;
        double truth_top_pt;
        double truth_top_eta;
        double truth_top_phi;
        double truth_top_e;

        double truth_tbar_mass;
        double truth_tbar_pt;
        double truth_tbar_eta;
        double truth_tbar_phi;
        double truth_tbar_e;

        double truth_top_neutino_mass;
        double truth_top_neutino_pt;
        double truth_top_neutino_eta;
        double truth_top_neutino_phi;
        double truth_top_neutino_e;

        double truth_tbar_neutino_mass;
        double truth_tbar_neutino_pt;
        double truth_tbar_neutino_eta;
        double truth_tbar_neutino_phi;
        double truth_tbar_neutino_e;

        double truth_top_neutrino_px;
        double truth_top_neutrino_py;
        double truth_top_neutrino_pz;
        double truth_tbar_neutrino_px;
        double truth_tbar_neutrino_py;
        double truth_tbar_neutrino_pz;

        double truth_tt_boost_parameter;

        bool save_nu_flows;
        bool save_initial_parton_info;

        bool branches_registered;

        double nu_flows_neutrino_p_x;
        double nu_flows_neutrino_p_y;
        double nu_flows_neutrino_p_z;

        double nu_flows_antineutrino_p_x;
        double nu_flows_antineutrino_p_y;
        double nu_flows_antineutrino_p_z;

        double nu_flows_m_ttbar;

        int truth_initial_parton_num_gluons;

        double dR_l1l2;

        int N_jets;

        EventType* Event;
};

#endif // PREPROCESSOR_H