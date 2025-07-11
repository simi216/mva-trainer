//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Thu Jun 26 18:12:29 2025 by ROOT version 6.36.00
// from TTree reco/xAOD->NTuple tree
// found on file: output_mc20_dilep_nuflows_partons.root
//////////////////////////////////////////////////////////

#ifndef reco_h
#define reco_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>


class reco {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
   Int_t           fCurrent; //!current Tree number in a TChain

// Fixed size dimensions of array or collections stored in the TTree if any.

   // Declaration of leaf types
   Int_t           PDFinfo_PDFID1;
   Int_t           PDFinfo_PDFID2;
   Int_t           PDFinfo_PDGID1;
   Int_t           PDFinfo_PDGID2;
   Float_t         PDFinfo_Q;
   Float_t         PDFinfo_X1;
   Float_t         PDFinfo_X2;
   Float_t         PDFinfo_XF1;
   Float_t         PDFinfo_XF2;
   Float_t         Ttbar_MC_W_afterFSR_from_t_eta;
   Float_t         Ttbar_MC_W_afterFSR_from_t_m;
   Int_t           Ttbar_MC_W_afterFSR_from_t_pdgId;
   Float_t         Ttbar_MC_W_afterFSR_from_t_phi;
   Float_t         Ttbar_MC_W_afterFSR_from_t_pt;
   Float_t         Ttbar_MC_W_afterFSR_from_tbar_eta;
   Float_t         Ttbar_MC_W_afterFSR_from_tbar_m;
   Int_t           Ttbar_MC_W_afterFSR_from_tbar_pdgId;
   Float_t         Ttbar_MC_W_afterFSR_from_tbar_phi;
   Float_t         Ttbar_MC_W_afterFSR_from_tbar_pt;
   Float_t         Ttbar_MC_W_beforeFSR_from_t_eta;
   Float_t         Ttbar_MC_W_beforeFSR_from_t_m;
   Int_t           Ttbar_MC_W_beforeFSR_from_t_pdgId;
   Float_t         Ttbar_MC_W_beforeFSR_from_t_phi;
   Float_t         Ttbar_MC_W_beforeFSR_from_t_pt;
   Float_t         Ttbar_MC_W_beforeFSR_from_tbar_eta;
   Float_t         Ttbar_MC_W_beforeFSR_from_tbar_m;
   Int_t           Ttbar_MC_W_beforeFSR_from_tbar_pdgId;
   Float_t         Ttbar_MC_W_beforeFSR_from_tbar_phi;
   Float_t         Ttbar_MC_W_beforeFSR_from_tbar_pt;
   Float_t         Ttbar_MC_Wdecay1_afterFSR_from_t_eta;
   Float_t         Ttbar_MC_Wdecay1_afterFSR_from_t_m;
   Int_t           Ttbar_MC_Wdecay1_afterFSR_from_t_pdgId;
   Float_t         Ttbar_MC_Wdecay1_afterFSR_from_t_phi;
   Float_t         Ttbar_MC_Wdecay1_afterFSR_from_t_pt;
   Float_t         Ttbar_MC_Wdecay1_afterFSR_from_tbar_eta;
   Float_t         Ttbar_MC_Wdecay1_afterFSR_from_tbar_m;
   Int_t           Ttbar_MC_Wdecay1_afterFSR_from_tbar_pdgId;
   Float_t         Ttbar_MC_Wdecay1_afterFSR_from_tbar_phi;
   Float_t         Ttbar_MC_Wdecay1_afterFSR_from_tbar_pt;
   Float_t         Ttbar_MC_Wdecay1_beforeFSR_from_t_eta;
   Float_t         Ttbar_MC_Wdecay1_beforeFSR_from_t_m;
   Int_t           Ttbar_MC_Wdecay1_beforeFSR_from_t_pdgId;
   Float_t         Ttbar_MC_Wdecay1_beforeFSR_from_t_phi;
   Float_t         Ttbar_MC_Wdecay1_beforeFSR_from_t_pt;
   Float_t         Ttbar_MC_Wdecay1_beforeFSR_from_tbar_eta;
   Float_t         Ttbar_MC_Wdecay1_beforeFSR_from_tbar_m;
   Int_t           Ttbar_MC_Wdecay1_beforeFSR_from_tbar_pdgId;
   Float_t         Ttbar_MC_Wdecay1_beforeFSR_from_tbar_phi;
   Float_t         Ttbar_MC_Wdecay1_beforeFSR_from_tbar_pt;
   Float_t         Ttbar_MC_Wdecay2_afterFSR_from_t_eta;
   Float_t         Ttbar_MC_Wdecay2_afterFSR_from_t_m;
   Int_t           Ttbar_MC_Wdecay2_afterFSR_from_t_pdgId;
   Float_t         Ttbar_MC_Wdecay2_afterFSR_from_t_phi;
   Float_t         Ttbar_MC_Wdecay2_afterFSR_from_t_pt;
   Float_t         Ttbar_MC_Wdecay2_afterFSR_from_tbar_eta;
   Float_t         Ttbar_MC_Wdecay2_afterFSR_from_tbar_m;
   Int_t           Ttbar_MC_Wdecay2_afterFSR_from_tbar_pdgId;
   Float_t         Ttbar_MC_Wdecay2_afterFSR_from_tbar_phi;
   Float_t         Ttbar_MC_Wdecay2_afterFSR_from_tbar_pt;
   Float_t         Ttbar_MC_Wdecay2_beforeFSR_from_t_eta;
   Float_t         Ttbar_MC_Wdecay2_beforeFSR_from_t_m;
   Int_t           Ttbar_MC_Wdecay2_beforeFSR_from_t_pdgId;
   Float_t         Ttbar_MC_Wdecay2_beforeFSR_from_t_phi;
   Float_t         Ttbar_MC_Wdecay2_beforeFSR_from_t_pt;
   Float_t         Ttbar_MC_Wdecay2_beforeFSR_from_tbar_eta;
   Float_t         Ttbar_MC_Wdecay2_beforeFSR_from_tbar_m;
   Int_t           Ttbar_MC_Wdecay2_beforeFSR_from_tbar_pdgId;
   Float_t         Ttbar_MC_Wdecay2_beforeFSR_from_tbar_phi;
   Float_t         Ttbar_MC_Wdecay2_beforeFSR_from_tbar_pt;
   Float_t         Ttbar_MC_b_afterFSR_from_t_eta;
   Float_t         Ttbar_MC_b_afterFSR_from_t_m;
   Int_t           Ttbar_MC_b_afterFSR_from_t_pdgId;
   Float_t         Ttbar_MC_b_afterFSR_from_t_phi;
   Float_t         Ttbar_MC_b_afterFSR_from_t_pt;
   Float_t         Ttbar_MC_b_beforeFSR_from_t_eta;
   Float_t         Ttbar_MC_b_beforeFSR_from_t_m;
   Int_t           Ttbar_MC_b_beforeFSR_from_t_pdgId;
   Float_t         Ttbar_MC_b_beforeFSR_from_t_phi;
   Float_t         Ttbar_MC_b_beforeFSR_from_t_pt;
   Float_t         Ttbar_MC_bbar_afterFSR_from_tbar_eta;
   Float_t         Ttbar_MC_bbar_afterFSR_from_tbar_m;
   Int_t           Ttbar_MC_bbar_afterFSR_from_tbar_pdgId;
   Float_t         Ttbar_MC_bbar_afterFSR_from_tbar_phi;
   Float_t         Ttbar_MC_bbar_afterFSR_from_tbar_pt;
   Float_t         Ttbar_MC_bbar_beforeFSR_from_tbar_eta;
   Float_t         Ttbar_MC_bbar_beforeFSR_from_tbar_m;
   Int_t           Ttbar_MC_bbar_beforeFSR_from_tbar_pdgId;
   Float_t         Ttbar_MC_bbar_beforeFSR_from_tbar_phi;
   Float_t         Ttbar_MC_bbar_beforeFSR_from_tbar_pt;
   Float_t         Ttbar_MC_t_afterFSR_eta;
   Float_t         Ttbar_MC_t_afterFSR_m;
   Int_t           Ttbar_MC_t_afterFSR_pdgId;
   Float_t         Ttbar_MC_t_afterFSR_phi;
   Float_t         Ttbar_MC_t_afterFSR_pt;
   Float_t         Ttbar_MC_t_beforeFSR_eta;
   Float_t         Ttbar_MC_t_beforeFSR_m;
   Int_t           Ttbar_MC_t_beforeFSR_pdgId;
   Float_t         Ttbar_MC_t_beforeFSR_phi;
   Float_t         Ttbar_MC_t_beforeFSR_pt;
   Float_t         Ttbar_MC_tbar_afterFSR_eta;
   Float_t         Ttbar_MC_tbar_afterFSR_m;
   Int_t           Ttbar_MC_tbar_afterFSR_pdgId;
   Float_t         Ttbar_MC_tbar_afterFSR_phi;
   Float_t         Ttbar_MC_tbar_afterFSR_pt;
   Float_t         Ttbar_MC_tbar_beforeFSR_eta;
   Float_t         Ttbar_MC_tbar_beforeFSR_m;
   Int_t           Ttbar_MC_tbar_beforeFSR_pdgId;
   Float_t         Ttbar_MC_tbar_beforeFSR_phi;
   Float_t         Ttbar_MC_tbar_beforeFSR_pt;
   Float_t         Ttbar_MC_ttbar_afterFSR_eta;
   Float_t         Ttbar_MC_ttbar_afterFSR_m;
   Float_t         Ttbar_MC_ttbar_afterFSR_phi;
   Float_t         Ttbar_MC_ttbar_afterFSR_pt;
   Float_t         Ttbar_MC_ttbar_beforeFSR_eta;
   Float_t         Ttbar_MC_ttbar_beforeFSR_m;
   Float_t         Ttbar_MC_ttbar_beforeFSR_phi;
   Float_t         Ttbar_MC_ttbar_beforeFSR_pt;
   Float_t         Ttbar_MC_ttbar_fromDecay_afterFSR_eta;
   Float_t         Ttbar_MC_ttbar_fromDecay_afterFSR_m;
   Float_t         Ttbar_MC_ttbar_fromDecay_afterFSR_phi;
   Float_t         Ttbar_MC_ttbar_fromDecay_afterFSR_pt;
   Float_t         Ttbar_MC_ttbar_fromDecay_beforeFSR_eta;
   Float_t         Ttbar_MC_ttbar_fromDecay_beforeFSR_m;
   Float_t         Ttbar_MC_ttbar_fromDecay_beforeFSR_phi;
   Float_t         Ttbar_MC_ttbar_fromDecay_beforeFSR_pt;
   Float_t         actualInteractionsPerCrossing;
   Float_t         averageInteractionsPerCrossing;
   std::vector<int>     *el_IFFClass;
   std::vector<float>   *el_charge;
   std::vector<float>   *el_eta;
   std::vector<float>   *el_phi;
   ULong64_t       eventNumber;
   std::vector<int>     *event_electron_truth_candidates;
   std::vector<int>     *event_electron_truth_idx;
   std::vector<int>     *event_jet_truth_candidates;
   std::vector<int>     *event_jet_truth_idx;
   std::vector<int>     *event_muon_truth_candidates;
   std::vector<int>     *event_muon_truth_idx;
   Int_t           event_nLeptons;
   std::vector<int>     *jet_GN2v01_Continuous_quantile;
   std::vector<float>   *jet_eta;
   std::vector<float>   *jet_phi;
   UInt_t          mcChannelNumber;
   std::vector<int>     *mu_IFFClass;
   std::vector<float>   *mu_charge;
   std::vector<float>   *mu_eta;
   std::vector<float>   *mu_phi;
   UInt_t          runNumber;
   Bool_t          trigPassed_HLT_e120_lhloose;
   Bool_t          trigPassed_HLT_e140_lhloose_L1EM22VHI;
   Bool_t          trigPassed_HLT_e140_lhloose_L1eEM26M;
   Bool_t          trigPassed_HLT_e140_lhloose_nod0;
   Bool_t          trigPassed_HLT_e24_lhmedium_L1EM20VH;
   Bool_t          trigPassed_HLT_e26_lhtight_ivarloose_L1EM22VHI;
   Bool_t          trigPassed_HLT_e26_lhtight_ivarloose_L1eEM26M;
   Bool_t          trigPassed_HLT_e26_lhtight_nod0_ivarloose;
   Bool_t          trigPassed_HLT_e60_lhmedium;
   Bool_t          trigPassed_HLT_e60_lhmedium_L1EM22VHI;
   Bool_t          trigPassed_HLT_e60_lhmedium_L1eEM26M;
   Bool_t          trigPassed_HLT_e60_lhmedium_nod0;
   Bool_t          trigPassed_HLT_mu20_iloose_L1MU15;
   Bool_t          trigPassed_HLT_mu24_ivarmedium_L1MU14FCH;
   Bool_t          trigPassed_HLT_mu26_ivarmedium;
   Bool_t          trigPassed_HLT_mu50;
   Bool_t          trigPassed_HLT_mu50_L1MU14FCH;
   Float_t         weight_beamspot;
   std::vector<float>   *el_e_NOSYS;
   std::vector<float>   *el_pt_NOSYS;
   std::vector<char>    *el_select_tight_NOSYS;
   Float_t         globalTriggerEffSF_NOSYS;
   Char_t          globalTriggerMatch_NOSYS;
   std::vector<float>   *jet_e_NOSYS;
   std::vector<float>   *jet_jvtEfficiency_NOSYS;
   std::vector<float>   *jet_pt_NOSYS;
   std::vector<char>    *jet_select_baselineJvt_NOSYS;
   std::vector<float>   *mu_TTVA_effSF_tight_NOSYS;
   std::vector<float>   *mu_e_NOSYS;
   std::vector<float>   *mu_isol_effSF_tight_NOSYS;
   std::vector<float>   *mu_pt_NOSYS;
   std::vector<float>   *mu_reco_effSF_tight_NOSYS;
   std::vector<char>    *mu_select_tight_NOSYS;
   Float_t         nuflows_loglik_NOSYS;
   std::vector<float>   *nuflows_nu_out_NOSYS;
   Char_t          pass_lljets_NOSYS;
   Float_t         weight_ftag_effSF_GN2v01_Continuous_NOSYS;
   Float_t         weight_jvt_effSF_NOSYS;
   Float_t         weight_leptonSF_tight_NOSYS;
   Float_t         weight_mc_NOSYS;
   Float_t         weight_pileup_NOSYS;
   Float_t         met_met_NOSYS;
   Float_t         met_phi_NOSYS;
   Float_t         met_significance_NOSYS;
   Float_t         met_sumet_NOSYS;

   // List of branches
   TBranch        *b_PDFinfo_PDFID1;   //!
   TBranch        *b_PDFinfo_PDFID2;   //!
   TBranch        *b_PDFinfo_PDGID1;   //!
   TBranch        *b_PDFinfo_PDGID2;   //!
   TBranch        *b_PDFinfo_Q;   //!
   TBranch        *b_PDFinfo_X1;   //!
   TBranch        *b_PDFinfo_X2;   //!
   TBranch        *b_PDFinfo_XF1;   //!
   TBranch        *b_PDFinfo_XF2;   //!
   TBranch        *b_Ttbar_MC_W_afterFSR_from_t_eta;   //!
   TBranch        *b_Ttbar_MC_W_afterFSR_from_t_m;   //!
   TBranch        *b_Ttbar_MC_W_afterFSR_from_t_pdgId;   //!
   TBranch        *b_Ttbar_MC_W_afterFSR_from_t_phi;   //!
   TBranch        *b_Ttbar_MC_W_afterFSR_from_t_pt;   //!
   TBranch        *b_Ttbar_MC_W_afterFSR_from_tbar_eta;   //!
   TBranch        *b_Ttbar_MC_W_afterFSR_from_tbar_m;   //!
   TBranch        *b_Ttbar_MC_W_afterFSR_from_tbar_pdgId;   //!
   TBranch        *b_Ttbar_MC_W_afterFSR_from_tbar_phi;   //!
   TBranch        *b_Ttbar_MC_W_afterFSR_from_tbar_pt;   //!
   TBranch        *b_Ttbar_MC_W_beforeFSR_from_t_eta;   //!
   TBranch        *b_Ttbar_MC_W_beforeFSR_from_t_m;   //!
   TBranch        *b_Ttbar_MC_W_beforeFSR_from_t_pdgId;   //!
   TBranch        *b_Ttbar_MC_W_beforeFSR_from_t_phi;   //!
   TBranch        *b_Ttbar_MC_W_beforeFSR_from_t_pt;   //!
   TBranch        *b_Ttbar_MC_W_beforeFSR_from_tbar_eta;   //!
   TBranch        *b_Ttbar_MC_W_beforeFSR_from_tbar_m;   //!
   TBranch        *b_Ttbar_MC_W_beforeFSR_from_tbar_pdgId;   //!
   TBranch        *b_Ttbar_MC_W_beforeFSR_from_tbar_phi;   //!
   TBranch        *b_Ttbar_MC_W_beforeFSR_from_tbar_pt;   //!
   TBranch        *b_Ttbar_MC_Wdecay1_afterFSR_from_t_eta;   //!
   TBranch        *b_Ttbar_MC_Wdecay1_afterFSR_from_t_m;   //!
   TBranch        *b_Ttbar_MC_Wdecay1_afterFSR_from_t_pdgId;   //!
   TBranch        *b_Ttbar_MC_Wdecay1_afterFSR_from_t_phi;   //!
   TBranch        *b_Ttbar_MC_Wdecay1_afterFSR_from_t_pt;   //!
   TBranch        *b_Ttbar_MC_Wdecay1_afterFSR_from_tbar_eta;   //!
   TBranch        *b_Ttbar_MC_Wdecay1_afterFSR_from_tbar_m;   //!
   TBranch        *b_Ttbar_MC_Wdecay1_afterFSR_from_tbar_pdgId;   //!
   TBranch        *b_Ttbar_MC_Wdecay1_afterFSR_from_tbar_phi;   //!
   TBranch        *b_Ttbar_MC_Wdecay1_afterFSR_from_tbar_pt;   //!
   TBranch        *b_Ttbar_MC_Wdecay1_beforeFSR_from_t_eta;   //!
   TBranch        *b_Ttbar_MC_Wdecay1_beforeFSR_from_t_m;   //!
   TBranch        *b_Ttbar_MC_Wdecay1_beforeFSR_from_t_pdgId;   //!
   TBranch        *b_Ttbar_MC_Wdecay1_beforeFSR_from_t_phi;   //!
   TBranch        *b_Ttbar_MC_Wdecay1_beforeFSR_from_t_pt;   //!
   TBranch        *b_Ttbar_MC_Wdecay1_beforeFSR_from_tbar_eta;   //!
   TBranch        *b_Ttbar_MC_Wdecay1_beforeFSR_from_tbar_m;   //!
   TBranch        *b_Ttbar_MC_Wdecay1_beforeFSR_from_tbar_pdgId;   //!
   TBranch        *b_Ttbar_MC_Wdecay1_beforeFSR_from_tbar_phi;   //!
   TBranch        *b_Ttbar_MC_Wdecay1_beforeFSR_from_tbar_pt;   //!
   TBranch        *b_Ttbar_MC_Wdecay2_afterFSR_from_t_eta;   //!
   TBranch        *b_Ttbar_MC_Wdecay2_afterFSR_from_t_m;   //!
   TBranch        *b_Ttbar_MC_Wdecay2_afterFSR_from_t_pdgId;   //!
   TBranch        *b_Ttbar_MC_Wdecay2_afterFSR_from_t_phi;   //!
   TBranch        *b_Ttbar_MC_Wdecay2_afterFSR_from_t_pt;   //!
   TBranch        *b_Ttbar_MC_Wdecay2_afterFSR_from_tbar_eta;   //!
   TBranch        *b_Ttbar_MC_Wdecay2_afterFSR_from_tbar_m;   //!
   TBranch        *b_Ttbar_MC_Wdecay2_afterFSR_from_tbar_pdgId;   //!
   TBranch        *b_Ttbar_MC_Wdecay2_afterFSR_from_tbar_phi;   //!
   TBranch        *b_Ttbar_MC_Wdecay2_afterFSR_from_tbar_pt;   //!
   TBranch        *b_Ttbar_MC_Wdecay2_beforeFSR_from_t_eta;   //!
   TBranch        *b_Ttbar_MC_Wdecay2_beforeFSR_from_t_m;   //!
   TBranch        *b_Ttbar_MC_Wdecay2_beforeFSR_from_t_pdgId;   //!
   TBranch        *b_Ttbar_MC_Wdecay2_beforeFSR_from_t_phi;   //!
   TBranch        *b_Ttbar_MC_Wdecay2_beforeFSR_from_t_pt;   //!
   TBranch        *b_Ttbar_MC_Wdecay2_beforeFSR_from_tbar_eta;   //!
   TBranch        *b_Ttbar_MC_Wdecay2_beforeFSR_from_tbar_m;   //!
   TBranch        *b_Ttbar_MC_Wdecay2_beforeFSR_from_tbar_pdgId;   //!
   TBranch        *b_Ttbar_MC_Wdecay2_beforeFSR_from_tbar_phi;   //!
   TBranch        *b_Ttbar_MC_Wdecay2_beforeFSR_from_tbar_pt;   //!
   TBranch        *b_Ttbar_MC_b_afterFSR_from_t_eta;   //!
   TBranch        *b_Ttbar_MC_b_afterFSR_from_t_m;   //!
   TBranch        *b_Ttbar_MC_b_afterFSR_from_t_pdgId;   //!
   TBranch        *b_Ttbar_MC_b_afterFSR_from_t_phi;   //!
   TBranch        *b_Ttbar_MC_b_afterFSR_from_t_pt;   //!
   TBranch        *b_Ttbar_MC_b_beforeFSR_from_t_eta;   //!
   TBranch        *b_Ttbar_MC_b_beforeFSR_from_t_m;   //!
   TBranch        *b_Ttbar_MC_b_beforeFSR_from_t_pdgId;   //!
   TBranch        *b_Ttbar_MC_b_beforeFSR_from_t_phi;   //!
   TBranch        *b_Ttbar_MC_b_beforeFSR_from_t_pt;   //!
   TBranch        *b_Ttbar_MC_bbar_afterFSR_from_tbar_eta;   //!
   TBranch        *b_Ttbar_MC_bbar_afterFSR_from_tbar_m;   //!
   TBranch        *b_Ttbar_MC_bbar_afterFSR_from_tbar_pdgId;   //!
   TBranch        *b_Ttbar_MC_bbar_afterFSR_from_tbar_phi;   //!
   TBranch        *b_Ttbar_MC_bbar_afterFSR_from_tbar_pt;   //!
   TBranch        *b_Ttbar_MC_bbar_beforeFSR_from_tbar_eta;   //!
   TBranch        *b_Ttbar_MC_bbar_beforeFSR_from_tbar_m;   //!
   TBranch        *b_Ttbar_MC_bbar_beforeFSR_from_tbar_pdgId;   //!
   TBranch        *b_Ttbar_MC_bbar_beforeFSR_from_tbar_phi;   //!
   TBranch        *b_Ttbar_MC_bbar_beforeFSR_from_tbar_pt;   //!
   TBranch        *b_Ttbar_MC_t_afterFSR_eta;   //!
   TBranch        *b_Ttbar_MC_t_afterFSR_m;   //!
   TBranch        *b_Ttbar_MC_t_afterFSR_pdgId;   //!
   TBranch        *b_Ttbar_MC_t_afterFSR_phi;   //!
   TBranch        *b_Ttbar_MC_t_afterFSR_pt;   //!
   TBranch        *b_Ttbar_MC_t_beforeFSR_eta;   //!
   TBranch        *b_Ttbar_MC_t_beforeFSR_m;   //!
   TBranch        *b_Ttbar_MC_t_beforeFSR_pdgId;   //!
   TBranch        *b_Ttbar_MC_t_beforeFSR_phi;   //!
   TBranch        *b_Ttbar_MC_t_beforeFSR_pt;   //!
   TBranch        *b_Ttbar_MC_tbar_afterFSR_eta;   //!
   TBranch        *b_Ttbar_MC_tbar_afterFSR_m;   //!
   TBranch        *b_Ttbar_MC_tbar_afterFSR_pdgId;   //!
   TBranch        *b_Ttbar_MC_tbar_afterFSR_phi;   //!
   TBranch        *b_Ttbar_MC_tbar_afterFSR_pt;   //!
   TBranch        *b_Ttbar_MC_tbar_beforeFSR_eta;   //!
   TBranch        *b_Ttbar_MC_tbar_beforeFSR_m;   //!
   TBranch        *b_Ttbar_MC_tbar_beforeFSR_pdgId;   //!
   TBranch        *b_Ttbar_MC_tbar_beforeFSR_phi;   //!
   TBranch        *b_Ttbar_MC_tbar_beforeFSR_pt;   //!
   TBranch        *b_Ttbar_MC_ttbar_afterFSR_eta;   //!
   TBranch        *b_Ttbar_MC_ttbar_afterFSR_m;   //!
   TBranch        *b_Ttbar_MC_ttbar_afterFSR_phi;   //!
   TBranch        *b_Ttbar_MC_ttbar_afterFSR_pt;   //!
   TBranch        *b_Ttbar_MC_ttbar_beforeFSR_eta;   //!
   TBranch        *b_Ttbar_MC_ttbar_beforeFSR_m;   //!
   TBranch        *b_Ttbar_MC_ttbar_beforeFSR_phi;   //!
   TBranch        *b_Ttbar_MC_ttbar_beforeFSR_pt;   //!
   TBranch        *b_Ttbar_MC_ttbar_fromDecay_afterFSR_eta;   //!
   TBranch        *b_Ttbar_MC_ttbar_fromDecay_afterFSR_m;   //!
   TBranch        *b_Ttbar_MC_ttbar_fromDecay_afterFSR_phi;   //!
   TBranch        *b_Ttbar_MC_ttbar_fromDecay_afterFSR_pt;   //!
   TBranch        *b_Ttbar_MC_ttbar_fromDecay_beforeFSR_eta;   //!
   TBranch        *b_Ttbar_MC_ttbar_fromDecay_beforeFSR_m;   //!
   TBranch        *b_Ttbar_MC_ttbar_fromDecay_beforeFSR_phi;   //!
   TBranch        *b_Ttbar_MC_ttbar_fromDecay_beforeFSR_pt;   //!
   TBranch        *b_actualInteractionsPerCrossing;   //!
   TBranch        *b_averageInteractionsPerCrossing;   //!
   TBranch        *b_el_IFFClass;   //!
   TBranch        *b_el_charge;   //!
   TBranch        *b_el_eta;   //!
   TBranch        *b_el_phi;   //!
   TBranch        *b_eventNumber;   //!
   TBranch        *b_event_electron_truth_candidates;   //!
   TBranch        *b_event_electron_truth_idx;   //!
   TBranch        *b_event_jet_truth_candidates;   //!
   TBranch        *b_event_jet_truth_idx;   //!
   TBranch        *b_event_muon_truth_candidates;   //!
   TBranch        *b_event_muon_truth_idx;   //!
   TBranch        *b_event_nLeptons;   //!
   TBranch        *b_jet_GN2v01_Continuous_quantile;   //!
   TBranch        *b_jet_eta;   //!
   TBranch        *b_jet_phi;   //!
   TBranch        *b_mcChannelNumber;   //!
   TBranch        *b_mu_IFFClass;   //!
   TBranch        *b_mu_charge;   //!
   TBranch        *b_mu_eta;   //!
   TBranch        *b_mu_phi;   //!
   TBranch        *b_runNumber;   //!
   TBranch        *b_trigPassed_HLT_e120_lhloose;   //!
   TBranch        *b_trigPassed_HLT_e140_lhloose_L1EM22VHI;   //!
   TBranch        *b_trigPassed_HLT_e140_lhloose_L1eEM26M;   //!
   TBranch        *b_trigPassed_HLT_e140_lhloose_nod0;   //!
   TBranch        *b_trigPassed_HLT_e24_lhmedium_L1EM20VH;   //!
   TBranch        *b_trigPassed_HLT_e26_lhtight_ivarloose_L1EM22VHI;   //!
   TBranch        *b_trigPassed_HLT_e26_lhtight_ivarloose_L1eEM26M;   //!
   TBranch        *b_trigPassed_HLT_e26_lhtight_nod0_ivarloose;   //!
   TBranch        *b_trigPassed_HLT_e60_lhmedium;   //!
   TBranch        *b_trigPassed_HLT_e60_lhmedium_L1EM22VHI;   //!
   TBranch        *b_trigPassed_HLT_e60_lhmedium_L1eEM26M;   //!
   TBranch        *b_trigPassed_HLT_e60_lhmedium_nod0;   //!
   TBranch        *b_trigPassed_HLT_mu20_iloose_L1MU15;   //!
   TBranch        *b_trigPassed_HLT_mu24_ivarmedium_L1MU14FCH;   //!
   TBranch        *b_trigPassed_HLT_mu26_ivarmedium;   //!
   TBranch        *b_trigPassed_HLT_mu50;   //!
   TBranch        *b_trigPassed_HLT_mu50_L1MU14FCH;   //!
   TBranch        *b_weight_beamspot;   //!
   TBranch        *b_el_e_NOSYS;   //!
   TBranch        *b_el_pt_NOSYS;   //!
   TBranch        *b_el_select_tight_NOSYS;   //!
   TBranch        *b_globalTriggerEffSF_NOSYS;   //!
   TBranch        *b_globalTriggerMatch_NOSYS;   //!
   TBranch        *b_jet_e_NOSYS;   //!
   TBranch        *b_jet_jvtEfficiency_NOSYS;   //!
   TBranch        *b_jet_pt_NOSYS;   //!
   TBranch        *b_jet_select_baselineJvt_NOSYS;   //!
   TBranch        *b_mu_TTVA_effSF_tight_NOSYS;   //!
   TBranch        *b_mu_e_NOSYS;   //!
   TBranch        *b_mu_isol_effSF_tight_NOSYS;   //!
   TBranch        *b_mu_pt_NOSYS;   //!
   TBranch        *b_mu_reco_effSF_tight_NOSYS;   //!
   TBranch        *b_mu_select_tight_NOSYS;   //!
   TBranch        *b_nuflows_loglik_NOSYS;   //!
   TBranch        *b_nuflows_nu_out_NOSYS;   //!
   TBranch        *b_pass_lljets_NOSYS;   //!
   TBranch        *b_weight_ftag_effSF_GN2v01_Continuous_NOSYS;   //!
   TBranch        *b_weight_jvt_effSF_NOSYS;   //!
   TBranch        *b_weight_leptonSF_tight_NOSYS;   //!
   TBranch        *b_weight_mc_NOSYS;   //!
   TBranch        *b_weight_pileup_NOSYS;   //!
   TBranch        *b_met_met_NOSYS;   //!
   TBranch        *b_met_phi_NOSYS;   //!
   TBranch        *b_met_significance_NOSYS;   //!
   TBranch        *b_met_sumet_NOSYS;   //!

   reco(TTree *tree=0);
   virtual ~reco();
   virtual Int_t    Cut(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry);
   virtual Long64_t LoadTree(Long64_t entry);
   virtual void     Init(TTree *tree);
   virtual void     Loop();
   virtual bool     Notify();
   virtual void     Show(Long64_t entry = -1);
};

#endif

#ifdef reco_cxx
reco::reco(TTree *tree) : fChain(0) 
{
// if parameter tree is not specified (or zero), connect the file
// used to generate this class and read the Tree.
   if (tree == 0) {
      TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("output_mc20_dilep_nuflows_partons.root");
      if (!f || !f->IsOpen()) {
         f = new TFile("output_mc20_dilep_nuflows_partons.root");
      }
      f->GetObject("reco",tree);

   }
   Init(tree);
}

reco::~reco()
{
   if (!fChain) return;
   delete fChain->GetCurrentFile();
}

Int_t reco::GetEntry(Long64_t entry)
{
// Read contents of entry.
   if (!fChain) return 0;
   return fChain->GetEntry(entry);
}
Long64_t reco::LoadTree(Long64_t entry)
{
// Set the environment to read one entry
   if (!fChain) return -5;
   Long64_t centry = fChain->LoadTree(entry);
   if (centry < 0) return centry;
   if (fChain->GetTreeNumber() != fCurrent) {
      fCurrent = fChain->GetTreeNumber();
      Notify();
   }
   return centry;
}

void reco::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses and branch
   // pointers of the tree will be set.
   // It is normally not necessary to make changes to the generated
   // code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running on PROOF
   // (once per file to be processed).

   // Set object pointer
   el_IFFClass = 0;
   el_charge = 0;
   el_eta = 0;
   el_phi = 0;
   event_electron_truth_candidates = 0;
   event_electron_truth_idx = 0;
   event_jet_truth_candidates = 0;
   event_jet_truth_idx = 0;
   event_muon_truth_candidates = 0;
   event_muon_truth_idx = 0;
   jet_GN2v01_Continuous_quantile = 0;
   jet_eta = 0;
   jet_phi = 0;
   mu_IFFClass = 0;
   mu_charge = 0;
   mu_eta = 0;
   mu_phi = 0;
   el_e_NOSYS = 0;
   el_pt_NOSYS = 0;
   el_select_tight_NOSYS = 0;
   jet_e_NOSYS = 0;
   jet_jvtEfficiency_NOSYS = 0;
   jet_pt_NOSYS = 0;
   jet_select_baselineJvt_NOSYS = 0;
   mu_TTVA_effSF_tight_NOSYS = 0;
   mu_e_NOSYS = 0;
   mu_isol_effSF_tight_NOSYS = 0;
   mu_pt_NOSYS = 0;
   mu_reco_effSF_tight_NOSYS = 0;
   mu_select_tight_NOSYS = 0;
   nuflows_nu_out_NOSYS = 0;
   // Set branch addresses and branch pointers
   if (!tree) return;
   fChain = tree;
   fCurrent = -1;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("PDFinfo_PDFID1", &PDFinfo_PDFID1, &b_PDFinfo_PDFID1);
   fChain->SetBranchAddress("PDFinfo_PDFID2", &PDFinfo_PDFID2, &b_PDFinfo_PDFID2);
   fChain->SetBranchAddress("PDFinfo_PDGID1", &PDFinfo_PDGID1, &b_PDFinfo_PDGID1);
   fChain->SetBranchAddress("PDFinfo_PDGID2", &PDFinfo_PDGID2, &b_PDFinfo_PDGID2);
   fChain->SetBranchAddress("PDFinfo_Q", &PDFinfo_Q, &b_PDFinfo_Q);
   fChain->SetBranchAddress("PDFinfo_X1", &PDFinfo_X1, &b_PDFinfo_X1);
   fChain->SetBranchAddress("PDFinfo_X2", &PDFinfo_X2, &b_PDFinfo_X2);
   fChain->SetBranchAddress("PDFinfo_XF1", &PDFinfo_XF1, &b_PDFinfo_XF1);
   fChain->SetBranchAddress("PDFinfo_XF2", &PDFinfo_XF2, &b_PDFinfo_XF2);
   fChain->SetBranchAddress("Ttbar_MC_W_afterFSR_from_t_eta", &Ttbar_MC_W_afterFSR_from_t_eta, &b_Ttbar_MC_W_afterFSR_from_t_eta);
   fChain->SetBranchAddress("Ttbar_MC_W_afterFSR_from_t_m", &Ttbar_MC_W_afterFSR_from_t_m, &b_Ttbar_MC_W_afterFSR_from_t_m);
   fChain->SetBranchAddress("Ttbar_MC_W_afterFSR_from_t_pdgId", &Ttbar_MC_W_afterFSR_from_t_pdgId, &b_Ttbar_MC_W_afterFSR_from_t_pdgId);
   fChain->SetBranchAddress("Ttbar_MC_W_afterFSR_from_t_phi", &Ttbar_MC_W_afterFSR_from_t_phi, &b_Ttbar_MC_W_afterFSR_from_t_phi);
   fChain->SetBranchAddress("Ttbar_MC_W_afterFSR_from_t_pt", &Ttbar_MC_W_afterFSR_from_t_pt, &b_Ttbar_MC_W_afterFSR_from_t_pt);
   fChain->SetBranchAddress("Ttbar_MC_W_afterFSR_from_tbar_eta", &Ttbar_MC_W_afterFSR_from_tbar_eta, &b_Ttbar_MC_W_afterFSR_from_tbar_eta);
   fChain->SetBranchAddress("Ttbar_MC_W_afterFSR_from_tbar_m", &Ttbar_MC_W_afterFSR_from_tbar_m, &b_Ttbar_MC_W_afterFSR_from_tbar_m);
   fChain->SetBranchAddress("Ttbar_MC_W_afterFSR_from_tbar_pdgId", &Ttbar_MC_W_afterFSR_from_tbar_pdgId, &b_Ttbar_MC_W_afterFSR_from_tbar_pdgId);
   fChain->SetBranchAddress("Ttbar_MC_W_afterFSR_from_tbar_phi", &Ttbar_MC_W_afterFSR_from_tbar_phi, &b_Ttbar_MC_W_afterFSR_from_tbar_phi);
   fChain->SetBranchAddress("Ttbar_MC_W_afterFSR_from_tbar_pt", &Ttbar_MC_W_afterFSR_from_tbar_pt, &b_Ttbar_MC_W_afterFSR_from_tbar_pt);
   fChain->SetBranchAddress("Ttbar_MC_W_beforeFSR_from_t_eta", &Ttbar_MC_W_beforeFSR_from_t_eta, &b_Ttbar_MC_W_beforeFSR_from_t_eta);
   fChain->SetBranchAddress("Ttbar_MC_W_beforeFSR_from_t_m", &Ttbar_MC_W_beforeFSR_from_t_m, &b_Ttbar_MC_W_beforeFSR_from_t_m);
   fChain->SetBranchAddress("Ttbar_MC_W_beforeFSR_from_t_pdgId", &Ttbar_MC_W_beforeFSR_from_t_pdgId, &b_Ttbar_MC_W_beforeFSR_from_t_pdgId);
   fChain->SetBranchAddress("Ttbar_MC_W_beforeFSR_from_t_phi", &Ttbar_MC_W_beforeFSR_from_t_phi, &b_Ttbar_MC_W_beforeFSR_from_t_phi);
   fChain->SetBranchAddress("Ttbar_MC_W_beforeFSR_from_t_pt", &Ttbar_MC_W_beforeFSR_from_t_pt, &b_Ttbar_MC_W_beforeFSR_from_t_pt);
   fChain->SetBranchAddress("Ttbar_MC_W_beforeFSR_from_tbar_eta", &Ttbar_MC_W_beforeFSR_from_tbar_eta, &b_Ttbar_MC_W_beforeFSR_from_tbar_eta);
   fChain->SetBranchAddress("Ttbar_MC_W_beforeFSR_from_tbar_m", &Ttbar_MC_W_beforeFSR_from_tbar_m, &b_Ttbar_MC_W_beforeFSR_from_tbar_m);
   fChain->SetBranchAddress("Ttbar_MC_W_beforeFSR_from_tbar_pdgId", &Ttbar_MC_W_beforeFSR_from_tbar_pdgId, &b_Ttbar_MC_W_beforeFSR_from_tbar_pdgId);
   fChain->SetBranchAddress("Ttbar_MC_W_beforeFSR_from_tbar_phi", &Ttbar_MC_W_beforeFSR_from_tbar_phi, &b_Ttbar_MC_W_beforeFSR_from_tbar_phi);
   fChain->SetBranchAddress("Ttbar_MC_W_beforeFSR_from_tbar_pt", &Ttbar_MC_W_beforeFSR_from_tbar_pt, &b_Ttbar_MC_W_beforeFSR_from_tbar_pt);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay1_afterFSR_from_t_eta", &Ttbar_MC_Wdecay1_afterFSR_from_t_eta, &b_Ttbar_MC_Wdecay1_afterFSR_from_t_eta);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay1_afterFSR_from_t_m", &Ttbar_MC_Wdecay1_afterFSR_from_t_m, &b_Ttbar_MC_Wdecay1_afterFSR_from_t_m);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay1_afterFSR_from_t_pdgId", &Ttbar_MC_Wdecay1_afterFSR_from_t_pdgId, &b_Ttbar_MC_Wdecay1_afterFSR_from_t_pdgId);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay1_afterFSR_from_t_phi", &Ttbar_MC_Wdecay1_afterFSR_from_t_phi, &b_Ttbar_MC_Wdecay1_afterFSR_from_t_phi);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay1_afterFSR_from_t_pt", &Ttbar_MC_Wdecay1_afterFSR_from_t_pt, &b_Ttbar_MC_Wdecay1_afterFSR_from_t_pt);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay1_afterFSR_from_tbar_eta", &Ttbar_MC_Wdecay1_afterFSR_from_tbar_eta, &b_Ttbar_MC_Wdecay1_afterFSR_from_tbar_eta);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay1_afterFSR_from_tbar_m", &Ttbar_MC_Wdecay1_afterFSR_from_tbar_m, &b_Ttbar_MC_Wdecay1_afterFSR_from_tbar_m);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay1_afterFSR_from_tbar_pdgId", &Ttbar_MC_Wdecay1_afterFSR_from_tbar_pdgId, &b_Ttbar_MC_Wdecay1_afterFSR_from_tbar_pdgId);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay1_afterFSR_from_tbar_phi", &Ttbar_MC_Wdecay1_afterFSR_from_tbar_phi, &b_Ttbar_MC_Wdecay1_afterFSR_from_tbar_phi);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay1_afterFSR_from_tbar_pt", &Ttbar_MC_Wdecay1_afterFSR_from_tbar_pt, &b_Ttbar_MC_Wdecay1_afterFSR_from_tbar_pt);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay1_beforeFSR_from_t_eta", &Ttbar_MC_Wdecay1_beforeFSR_from_t_eta, &b_Ttbar_MC_Wdecay1_beforeFSR_from_t_eta);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay1_beforeFSR_from_t_m", &Ttbar_MC_Wdecay1_beforeFSR_from_t_m, &b_Ttbar_MC_Wdecay1_beforeFSR_from_t_m);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay1_beforeFSR_from_t_pdgId", &Ttbar_MC_Wdecay1_beforeFSR_from_t_pdgId, &b_Ttbar_MC_Wdecay1_beforeFSR_from_t_pdgId);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay1_beforeFSR_from_t_phi", &Ttbar_MC_Wdecay1_beforeFSR_from_t_phi, &b_Ttbar_MC_Wdecay1_beforeFSR_from_t_phi);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay1_beforeFSR_from_t_pt", &Ttbar_MC_Wdecay1_beforeFSR_from_t_pt, &b_Ttbar_MC_Wdecay1_beforeFSR_from_t_pt);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay1_beforeFSR_from_tbar_eta", &Ttbar_MC_Wdecay1_beforeFSR_from_tbar_eta, &b_Ttbar_MC_Wdecay1_beforeFSR_from_tbar_eta);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay1_beforeFSR_from_tbar_m", &Ttbar_MC_Wdecay1_beforeFSR_from_tbar_m, &b_Ttbar_MC_Wdecay1_beforeFSR_from_tbar_m);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay1_beforeFSR_from_tbar_pdgId", &Ttbar_MC_Wdecay1_beforeFSR_from_tbar_pdgId, &b_Ttbar_MC_Wdecay1_beforeFSR_from_tbar_pdgId);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay1_beforeFSR_from_tbar_phi", &Ttbar_MC_Wdecay1_beforeFSR_from_tbar_phi, &b_Ttbar_MC_Wdecay1_beforeFSR_from_tbar_phi);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay1_beforeFSR_from_tbar_pt", &Ttbar_MC_Wdecay1_beforeFSR_from_tbar_pt, &b_Ttbar_MC_Wdecay1_beforeFSR_from_tbar_pt);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay2_afterFSR_from_t_eta", &Ttbar_MC_Wdecay2_afterFSR_from_t_eta, &b_Ttbar_MC_Wdecay2_afterFSR_from_t_eta);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay2_afterFSR_from_t_m", &Ttbar_MC_Wdecay2_afterFSR_from_t_m, &b_Ttbar_MC_Wdecay2_afterFSR_from_t_m);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay2_afterFSR_from_t_pdgId", &Ttbar_MC_Wdecay2_afterFSR_from_t_pdgId, &b_Ttbar_MC_Wdecay2_afterFSR_from_t_pdgId);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay2_afterFSR_from_t_phi", &Ttbar_MC_Wdecay2_afterFSR_from_t_phi, &b_Ttbar_MC_Wdecay2_afterFSR_from_t_phi);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay2_afterFSR_from_t_pt", &Ttbar_MC_Wdecay2_afterFSR_from_t_pt, &b_Ttbar_MC_Wdecay2_afterFSR_from_t_pt);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay2_afterFSR_from_tbar_eta", &Ttbar_MC_Wdecay2_afterFSR_from_tbar_eta, &b_Ttbar_MC_Wdecay2_afterFSR_from_tbar_eta);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay2_afterFSR_from_tbar_m", &Ttbar_MC_Wdecay2_afterFSR_from_tbar_m, &b_Ttbar_MC_Wdecay2_afterFSR_from_tbar_m);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay2_afterFSR_from_tbar_pdgId", &Ttbar_MC_Wdecay2_afterFSR_from_tbar_pdgId, &b_Ttbar_MC_Wdecay2_afterFSR_from_tbar_pdgId);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay2_afterFSR_from_tbar_phi", &Ttbar_MC_Wdecay2_afterFSR_from_tbar_phi, &b_Ttbar_MC_Wdecay2_afterFSR_from_tbar_phi);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay2_afterFSR_from_tbar_pt", &Ttbar_MC_Wdecay2_afterFSR_from_tbar_pt, &b_Ttbar_MC_Wdecay2_afterFSR_from_tbar_pt);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay2_beforeFSR_from_t_eta", &Ttbar_MC_Wdecay2_beforeFSR_from_t_eta, &b_Ttbar_MC_Wdecay2_beforeFSR_from_t_eta);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay2_beforeFSR_from_t_m", &Ttbar_MC_Wdecay2_beforeFSR_from_t_m, &b_Ttbar_MC_Wdecay2_beforeFSR_from_t_m);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay2_beforeFSR_from_t_pdgId", &Ttbar_MC_Wdecay2_beforeFSR_from_t_pdgId, &b_Ttbar_MC_Wdecay2_beforeFSR_from_t_pdgId);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay2_beforeFSR_from_t_phi", &Ttbar_MC_Wdecay2_beforeFSR_from_t_phi, &b_Ttbar_MC_Wdecay2_beforeFSR_from_t_phi);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay2_beforeFSR_from_t_pt", &Ttbar_MC_Wdecay2_beforeFSR_from_t_pt, &b_Ttbar_MC_Wdecay2_beforeFSR_from_t_pt);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay2_beforeFSR_from_tbar_eta", &Ttbar_MC_Wdecay2_beforeFSR_from_tbar_eta, &b_Ttbar_MC_Wdecay2_beforeFSR_from_tbar_eta);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay2_beforeFSR_from_tbar_m", &Ttbar_MC_Wdecay2_beforeFSR_from_tbar_m, &b_Ttbar_MC_Wdecay2_beforeFSR_from_tbar_m);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay2_beforeFSR_from_tbar_pdgId", &Ttbar_MC_Wdecay2_beforeFSR_from_tbar_pdgId, &b_Ttbar_MC_Wdecay2_beforeFSR_from_tbar_pdgId);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay2_beforeFSR_from_tbar_phi", &Ttbar_MC_Wdecay2_beforeFSR_from_tbar_phi, &b_Ttbar_MC_Wdecay2_beforeFSR_from_tbar_phi);
   fChain->SetBranchAddress("Ttbar_MC_Wdecay2_beforeFSR_from_tbar_pt", &Ttbar_MC_Wdecay2_beforeFSR_from_tbar_pt, &b_Ttbar_MC_Wdecay2_beforeFSR_from_tbar_pt);
   fChain->SetBranchAddress("Ttbar_MC_b_afterFSR_from_t_eta", &Ttbar_MC_b_afterFSR_from_t_eta, &b_Ttbar_MC_b_afterFSR_from_t_eta);
   fChain->SetBranchAddress("Ttbar_MC_b_afterFSR_from_t_m", &Ttbar_MC_b_afterFSR_from_t_m, &b_Ttbar_MC_b_afterFSR_from_t_m);
   fChain->SetBranchAddress("Ttbar_MC_b_afterFSR_from_t_pdgId", &Ttbar_MC_b_afterFSR_from_t_pdgId, &b_Ttbar_MC_b_afterFSR_from_t_pdgId);
   fChain->SetBranchAddress("Ttbar_MC_b_afterFSR_from_t_phi", &Ttbar_MC_b_afterFSR_from_t_phi, &b_Ttbar_MC_b_afterFSR_from_t_phi);
   fChain->SetBranchAddress("Ttbar_MC_b_afterFSR_from_t_pt", &Ttbar_MC_b_afterFSR_from_t_pt, &b_Ttbar_MC_b_afterFSR_from_t_pt);
   fChain->SetBranchAddress("Ttbar_MC_b_beforeFSR_from_t_eta", &Ttbar_MC_b_beforeFSR_from_t_eta, &b_Ttbar_MC_b_beforeFSR_from_t_eta);
   fChain->SetBranchAddress("Ttbar_MC_b_beforeFSR_from_t_m", &Ttbar_MC_b_beforeFSR_from_t_m, &b_Ttbar_MC_b_beforeFSR_from_t_m);
   fChain->SetBranchAddress("Ttbar_MC_b_beforeFSR_from_t_pdgId", &Ttbar_MC_b_beforeFSR_from_t_pdgId, &b_Ttbar_MC_b_beforeFSR_from_t_pdgId);
   fChain->SetBranchAddress("Ttbar_MC_b_beforeFSR_from_t_phi", &Ttbar_MC_b_beforeFSR_from_t_phi, &b_Ttbar_MC_b_beforeFSR_from_t_phi);
   fChain->SetBranchAddress("Ttbar_MC_b_beforeFSR_from_t_pt", &Ttbar_MC_b_beforeFSR_from_t_pt, &b_Ttbar_MC_b_beforeFSR_from_t_pt);
   fChain->SetBranchAddress("Ttbar_MC_bbar_afterFSR_from_tbar_eta", &Ttbar_MC_bbar_afterFSR_from_tbar_eta, &b_Ttbar_MC_bbar_afterFSR_from_tbar_eta);
   fChain->SetBranchAddress("Ttbar_MC_bbar_afterFSR_from_tbar_m", &Ttbar_MC_bbar_afterFSR_from_tbar_m, &b_Ttbar_MC_bbar_afterFSR_from_tbar_m);
   fChain->SetBranchAddress("Ttbar_MC_bbar_afterFSR_from_tbar_pdgId", &Ttbar_MC_bbar_afterFSR_from_tbar_pdgId, &b_Ttbar_MC_bbar_afterFSR_from_tbar_pdgId);
   fChain->SetBranchAddress("Ttbar_MC_bbar_afterFSR_from_tbar_phi", &Ttbar_MC_bbar_afterFSR_from_tbar_phi, &b_Ttbar_MC_bbar_afterFSR_from_tbar_phi);
   fChain->SetBranchAddress("Ttbar_MC_bbar_afterFSR_from_tbar_pt", &Ttbar_MC_bbar_afterFSR_from_tbar_pt, &b_Ttbar_MC_bbar_afterFSR_from_tbar_pt);
   fChain->SetBranchAddress("Ttbar_MC_bbar_beforeFSR_from_tbar_eta", &Ttbar_MC_bbar_beforeFSR_from_tbar_eta, &b_Ttbar_MC_bbar_beforeFSR_from_tbar_eta);
   fChain->SetBranchAddress("Ttbar_MC_bbar_beforeFSR_from_tbar_m", &Ttbar_MC_bbar_beforeFSR_from_tbar_m, &b_Ttbar_MC_bbar_beforeFSR_from_tbar_m);
   fChain->SetBranchAddress("Ttbar_MC_bbar_beforeFSR_from_tbar_pdgId", &Ttbar_MC_bbar_beforeFSR_from_tbar_pdgId, &b_Ttbar_MC_bbar_beforeFSR_from_tbar_pdgId);
   fChain->SetBranchAddress("Ttbar_MC_bbar_beforeFSR_from_tbar_phi", &Ttbar_MC_bbar_beforeFSR_from_tbar_phi, &b_Ttbar_MC_bbar_beforeFSR_from_tbar_phi);
   fChain->SetBranchAddress("Ttbar_MC_bbar_beforeFSR_from_tbar_pt", &Ttbar_MC_bbar_beforeFSR_from_tbar_pt, &b_Ttbar_MC_bbar_beforeFSR_from_tbar_pt);
   fChain->SetBranchAddress("Ttbar_MC_t_afterFSR_eta", &Ttbar_MC_t_afterFSR_eta, &b_Ttbar_MC_t_afterFSR_eta);
   fChain->SetBranchAddress("Ttbar_MC_t_afterFSR_m", &Ttbar_MC_t_afterFSR_m, &b_Ttbar_MC_t_afterFSR_m);
   fChain->SetBranchAddress("Ttbar_MC_t_afterFSR_pdgId", &Ttbar_MC_t_afterFSR_pdgId, &b_Ttbar_MC_t_afterFSR_pdgId);
   fChain->SetBranchAddress("Ttbar_MC_t_afterFSR_phi", &Ttbar_MC_t_afterFSR_phi, &b_Ttbar_MC_t_afterFSR_phi);
   fChain->SetBranchAddress("Ttbar_MC_t_afterFSR_pt", &Ttbar_MC_t_afterFSR_pt, &b_Ttbar_MC_t_afterFSR_pt);
   fChain->SetBranchAddress("Ttbar_MC_t_beforeFSR_eta", &Ttbar_MC_t_beforeFSR_eta, &b_Ttbar_MC_t_beforeFSR_eta);
   fChain->SetBranchAddress("Ttbar_MC_t_beforeFSR_m", &Ttbar_MC_t_beforeFSR_m, &b_Ttbar_MC_t_beforeFSR_m);
   fChain->SetBranchAddress("Ttbar_MC_t_beforeFSR_pdgId", &Ttbar_MC_t_beforeFSR_pdgId, &b_Ttbar_MC_t_beforeFSR_pdgId);
   fChain->SetBranchAddress("Ttbar_MC_t_beforeFSR_phi", &Ttbar_MC_t_beforeFSR_phi, &b_Ttbar_MC_t_beforeFSR_phi);
   fChain->SetBranchAddress("Ttbar_MC_t_beforeFSR_pt", &Ttbar_MC_t_beforeFSR_pt, &b_Ttbar_MC_t_beforeFSR_pt);
   fChain->SetBranchAddress("Ttbar_MC_tbar_afterFSR_eta", &Ttbar_MC_tbar_afterFSR_eta, &b_Ttbar_MC_tbar_afterFSR_eta);
   fChain->SetBranchAddress("Ttbar_MC_tbar_afterFSR_m", &Ttbar_MC_tbar_afterFSR_m, &b_Ttbar_MC_tbar_afterFSR_m);
   fChain->SetBranchAddress("Ttbar_MC_tbar_afterFSR_pdgId", &Ttbar_MC_tbar_afterFSR_pdgId, &b_Ttbar_MC_tbar_afterFSR_pdgId);
   fChain->SetBranchAddress("Ttbar_MC_tbar_afterFSR_phi", &Ttbar_MC_tbar_afterFSR_phi, &b_Ttbar_MC_tbar_afterFSR_phi);
   fChain->SetBranchAddress("Ttbar_MC_tbar_afterFSR_pt", &Ttbar_MC_tbar_afterFSR_pt, &b_Ttbar_MC_tbar_afterFSR_pt);
   fChain->SetBranchAddress("Ttbar_MC_tbar_beforeFSR_eta", &Ttbar_MC_tbar_beforeFSR_eta, &b_Ttbar_MC_tbar_beforeFSR_eta);
   fChain->SetBranchAddress("Ttbar_MC_tbar_beforeFSR_m", &Ttbar_MC_tbar_beforeFSR_m, &b_Ttbar_MC_tbar_beforeFSR_m);
   fChain->SetBranchAddress("Ttbar_MC_tbar_beforeFSR_pdgId", &Ttbar_MC_tbar_beforeFSR_pdgId, &b_Ttbar_MC_tbar_beforeFSR_pdgId);
   fChain->SetBranchAddress("Ttbar_MC_tbar_beforeFSR_phi", &Ttbar_MC_tbar_beforeFSR_phi, &b_Ttbar_MC_tbar_beforeFSR_phi);
   fChain->SetBranchAddress("Ttbar_MC_tbar_beforeFSR_pt", &Ttbar_MC_tbar_beforeFSR_pt, &b_Ttbar_MC_tbar_beforeFSR_pt);
   fChain->SetBranchAddress("Ttbar_MC_ttbar_afterFSR_eta", &Ttbar_MC_ttbar_afterFSR_eta, &b_Ttbar_MC_ttbar_afterFSR_eta);
   fChain->SetBranchAddress("Ttbar_MC_ttbar_afterFSR_m", &Ttbar_MC_ttbar_afterFSR_m, &b_Ttbar_MC_ttbar_afterFSR_m);
   fChain->SetBranchAddress("Ttbar_MC_ttbar_afterFSR_phi", &Ttbar_MC_ttbar_afterFSR_phi, &b_Ttbar_MC_ttbar_afterFSR_phi);
   fChain->SetBranchAddress("Ttbar_MC_ttbar_afterFSR_pt", &Ttbar_MC_ttbar_afterFSR_pt, &b_Ttbar_MC_ttbar_afterFSR_pt);
   fChain->SetBranchAddress("Ttbar_MC_ttbar_beforeFSR_eta", &Ttbar_MC_ttbar_beforeFSR_eta, &b_Ttbar_MC_ttbar_beforeFSR_eta);
   fChain->SetBranchAddress("Ttbar_MC_ttbar_beforeFSR_m", &Ttbar_MC_ttbar_beforeFSR_m, &b_Ttbar_MC_ttbar_beforeFSR_m);
   fChain->SetBranchAddress("Ttbar_MC_ttbar_beforeFSR_phi", &Ttbar_MC_ttbar_beforeFSR_phi, &b_Ttbar_MC_ttbar_beforeFSR_phi);
   fChain->SetBranchAddress("Ttbar_MC_ttbar_beforeFSR_pt", &Ttbar_MC_ttbar_beforeFSR_pt, &b_Ttbar_MC_ttbar_beforeFSR_pt);
   fChain->SetBranchAddress("Ttbar_MC_ttbar_fromDecay_afterFSR_eta", &Ttbar_MC_ttbar_fromDecay_afterFSR_eta, &b_Ttbar_MC_ttbar_fromDecay_afterFSR_eta);
   fChain->SetBranchAddress("Ttbar_MC_ttbar_fromDecay_afterFSR_m", &Ttbar_MC_ttbar_fromDecay_afterFSR_m, &b_Ttbar_MC_ttbar_fromDecay_afterFSR_m);
   fChain->SetBranchAddress("Ttbar_MC_ttbar_fromDecay_afterFSR_phi", &Ttbar_MC_ttbar_fromDecay_afterFSR_phi, &b_Ttbar_MC_ttbar_fromDecay_afterFSR_phi);
   fChain->SetBranchAddress("Ttbar_MC_ttbar_fromDecay_afterFSR_pt", &Ttbar_MC_ttbar_fromDecay_afterFSR_pt, &b_Ttbar_MC_ttbar_fromDecay_afterFSR_pt);
   fChain->SetBranchAddress("Ttbar_MC_ttbar_fromDecay_beforeFSR_eta", &Ttbar_MC_ttbar_fromDecay_beforeFSR_eta, &b_Ttbar_MC_ttbar_fromDecay_beforeFSR_eta);
   fChain->SetBranchAddress("Ttbar_MC_ttbar_fromDecay_beforeFSR_m", &Ttbar_MC_ttbar_fromDecay_beforeFSR_m, &b_Ttbar_MC_ttbar_fromDecay_beforeFSR_m);
   fChain->SetBranchAddress("Ttbar_MC_ttbar_fromDecay_beforeFSR_phi", &Ttbar_MC_ttbar_fromDecay_beforeFSR_phi, &b_Ttbar_MC_ttbar_fromDecay_beforeFSR_phi);
   fChain->SetBranchAddress("Ttbar_MC_ttbar_fromDecay_beforeFSR_pt", &Ttbar_MC_ttbar_fromDecay_beforeFSR_pt, &b_Ttbar_MC_ttbar_fromDecay_beforeFSR_pt);
   fChain->SetBranchAddress("actualInteractionsPerCrossing", &actualInteractionsPerCrossing, &b_actualInteractionsPerCrossing);
   fChain->SetBranchAddress("averageInteractionsPerCrossing", &averageInteractionsPerCrossing, &b_averageInteractionsPerCrossing);
   fChain->SetBranchAddress("el_IFFClass", &el_IFFClass, &b_el_IFFClass);
   fChain->SetBranchAddress("el_charge", &el_charge, &b_el_charge);
   fChain->SetBranchAddress("el_eta", &el_eta, &b_el_eta);
   fChain->SetBranchAddress("el_phi", &el_phi, &b_el_phi);
   fChain->SetBranchAddress("eventNumber", &eventNumber, &b_eventNumber);
   fChain->SetBranchAddress("event_electron_truth_candidates", &event_electron_truth_candidates, &b_event_electron_truth_candidates);
   fChain->SetBranchAddress("event_electron_truth_idx", &event_electron_truth_idx, &b_event_electron_truth_idx);
   fChain->SetBranchAddress("event_jet_truth_candidates", &event_jet_truth_candidates, &b_event_jet_truth_candidates);
   fChain->SetBranchAddress("event_jet_truth_idx", &event_jet_truth_idx, &b_event_jet_truth_idx);
   fChain->SetBranchAddress("event_muon_truth_candidates", &event_muon_truth_candidates, &b_event_muon_truth_candidates);
   fChain->SetBranchAddress("event_muon_truth_idx", &event_muon_truth_idx, &b_event_muon_truth_idx);
   fChain->SetBranchAddress("event_nLeptons", &event_nLeptons, &b_event_nLeptons);
   fChain->SetBranchAddress("jet_GN2v01_Continuous_quantile", &jet_GN2v01_Continuous_quantile, &b_jet_GN2v01_Continuous_quantile);
   fChain->SetBranchAddress("jet_eta", &jet_eta, &b_jet_eta);
   fChain->SetBranchAddress("jet_phi", &jet_phi, &b_jet_phi);
   fChain->SetBranchAddress("mcChannelNumber", &mcChannelNumber, &b_mcChannelNumber);
   fChain->SetBranchAddress("mu_IFFClass", &mu_IFFClass, &b_mu_IFFClass);
   fChain->SetBranchAddress("mu_charge", &mu_charge, &b_mu_charge);
   fChain->SetBranchAddress("mu_eta", &mu_eta, &b_mu_eta);
   fChain->SetBranchAddress("mu_phi", &mu_phi, &b_mu_phi);
   fChain->SetBranchAddress("runNumber", &runNumber, &b_runNumber);
   fChain->SetBranchAddress("trigPassed_HLT_e120_lhloose", &trigPassed_HLT_e120_lhloose, &b_trigPassed_HLT_e120_lhloose);
   fChain->SetBranchAddress("trigPassed_HLT_e140_lhloose_L1EM22VHI", &trigPassed_HLT_e140_lhloose_L1EM22VHI, &b_trigPassed_HLT_e140_lhloose_L1EM22VHI);
   fChain->SetBranchAddress("trigPassed_HLT_e140_lhloose_L1eEM26M", &trigPassed_HLT_e140_lhloose_L1eEM26M, &b_trigPassed_HLT_e140_lhloose_L1eEM26M);
   fChain->SetBranchAddress("trigPassed_HLT_e140_lhloose_nod0", &trigPassed_HLT_e140_lhloose_nod0, &b_trigPassed_HLT_e140_lhloose_nod0);
   fChain->SetBranchAddress("trigPassed_HLT_e24_lhmedium_L1EM20VH", &trigPassed_HLT_e24_lhmedium_L1EM20VH, &b_trigPassed_HLT_e24_lhmedium_L1EM20VH);
   fChain->SetBranchAddress("trigPassed_HLT_e26_lhtight_ivarloose_L1EM22VHI", &trigPassed_HLT_e26_lhtight_ivarloose_L1EM22VHI, &b_trigPassed_HLT_e26_lhtight_ivarloose_L1EM22VHI);
   fChain->SetBranchAddress("trigPassed_HLT_e26_lhtight_ivarloose_L1eEM26M", &trigPassed_HLT_e26_lhtight_ivarloose_L1eEM26M, &b_trigPassed_HLT_e26_lhtight_ivarloose_L1eEM26M);
   fChain->SetBranchAddress("trigPassed_HLT_e26_lhtight_nod0_ivarloose", &trigPassed_HLT_e26_lhtight_nod0_ivarloose, &b_trigPassed_HLT_e26_lhtight_nod0_ivarloose);
   fChain->SetBranchAddress("trigPassed_HLT_e60_lhmedium", &trigPassed_HLT_e60_lhmedium, &b_trigPassed_HLT_e60_lhmedium);
   fChain->SetBranchAddress("trigPassed_HLT_e60_lhmedium_L1EM22VHI", &trigPassed_HLT_e60_lhmedium_L1EM22VHI, &b_trigPassed_HLT_e60_lhmedium_L1EM22VHI);
   fChain->SetBranchAddress("trigPassed_HLT_e60_lhmedium_L1eEM26M", &trigPassed_HLT_e60_lhmedium_L1eEM26M, &b_trigPassed_HLT_e60_lhmedium_L1eEM26M);
   fChain->SetBranchAddress("trigPassed_HLT_e60_lhmedium_nod0", &trigPassed_HLT_e60_lhmedium_nod0, &b_trigPassed_HLT_e60_lhmedium_nod0);
   fChain->SetBranchAddress("trigPassed_HLT_mu20_iloose_L1MU15", &trigPassed_HLT_mu20_iloose_L1MU15, &b_trigPassed_HLT_mu20_iloose_L1MU15);
   fChain->SetBranchAddress("trigPassed_HLT_mu24_ivarmedium_L1MU14FCH", &trigPassed_HLT_mu24_ivarmedium_L1MU14FCH, &b_trigPassed_HLT_mu24_ivarmedium_L1MU14FCH);
   fChain->SetBranchAddress("trigPassed_HLT_mu26_ivarmedium", &trigPassed_HLT_mu26_ivarmedium, &b_trigPassed_HLT_mu26_ivarmedium);
   fChain->SetBranchAddress("trigPassed_HLT_mu50", &trigPassed_HLT_mu50, &b_trigPassed_HLT_mu50);
   fChain->SetBranchAddress("trigPassed_HLT_mu50_L1MU14FCH", &trigPassed_HLT_mu50_L1MU14FCH, &b_trigPassed_HLT_mu50_L1MU14FCH);
   fChain->SetBranchAddress("weight_beamspot", &weight_beamspot, &b_weight_beamspot);
   fChain->SetBranchAddress("el_e_NOSYS", &el_e_NOSYS, &b_el_e_NOSYS);
   fChain->SetBranchAddress("el_pt_NOSYS", &el_pt_NOSYS, &b_el_pt_NOSYS);
   fChain->SetBranchAddress("el_select_tight_NOSYS", &el_select_tight_NOSYS, &b_el_select_tight_NOSYS);
   fChain->SetBranchAddress("globalTriggerEffSF_NOSYS", &globalTriggerEffSF_NOSYS, &b_globalTriggerEffSF_NOSYS);
   fChain->SetBranchAddress("globalTriggerMatch_NOSYS", &globalTriggerMatch_NOSYS, &b_globalTriggerMatch_NOSYS);
   fChain->SetBranchAddress("jet_e_NOSYS", &jet_e_NOSYS, &b_jet_e_NOSYS);
   fChain->SetBranchAddress("jet_jvtEfficiency_NOSYS", &jet_jvtEfficiency_NOSYS, &b_jet_jvtEfficiency_NOSYS);
   fChain->SetBranchAddress("jet_pt_NOSYS", &jet_pt_NOSYS, &b_jet_pt_NOSYS);
   fChain->SetBranchAddress("jet_select_baselineJvt_NOSYS", &jet_select_baselineJvt_NOSYS, &b_jet_select_baselineJvt_NOSYS);
   fChain->SetBranchAddress("mu_TTVA_effSF_tight_NOSYS", &mu_TTVA_effSF_tight_NOSYS, &b_mu_TTVA_effSF_tight_NOSYS);
   fChain->SetBranchAddress("mu_e_NOSYS", &mu_e_NOSYS, &b_mu_e_NOSYS);
   fChain->SetBranchAddress("mu_isol_effSF_tight_NOSYS", &mu_isol_effSF_tight_NOSYS, &b_mu_isol_effSF_tight_NOSYS);
   fChain->SetBranchAddress("mu_pt_NOSYS", &mu_pt_NOSYS, &b_mu_pt_NOSYS);
   fChain->SetBranchAddress("mu_reco_effSF_tight_NOSYS", &mu_reco_effSF_tight_NOSYS, &b_mu_reco_effSF_tight_NOSYS);
   fChain->SetBranchAddress("mu_select_tight_NOSYS", &mu_select_tight_NOSYS, &b_mu_select_tight_NOSYS);
   fChain->SetBranchAddress("nuflows_loglik_NOSYS", &nuflows_loglik_NOSYS, &b_nuflows_loglik_NOSYS);
   fChain->SetBranchAddress("nuflows_nu_out_NOSYS", &nuflows_nu_out_NOSYS, &b_nuflows_nu_out_NOSYS);
   fChain->SetBranchAddress("pass_lljets_NOSYS", &pass_lljets_NOSYS, &b_pass_lljets_NOSYS);
   fChain->SetBranchAddress("weight_ftag_effSF_GN2v01_Continuous_NOSYS", &weight_ftag_effSF_GN2v01_Continuous_NOSYS, &b_weight_ftag_effSF_GN2v01_Continuous_NOSYS);
   fChain->SetBranchAddress("weight_jvt_effSF_NOSYS", &weight_jvt_effSF_NOSYS, &b_weight_jvt_effSF_NOSYS);
   fChain->SetBranchAddress("weight_leptonSF_tight_NOSYS", &weight_leptonSF_tight_NOSYS, &b_weight_leptonSF_tight_NOSYS);
   fChain->SetBranchAddress("weight_mc_NOSYS", &weight_mc_NOSYS, &b_weight_mc_NOSYS);
   fChain->SetBranchAddress("weight_pileup_NOSYS", &weight_pileup_NOSYS, &b_weight_pileup_NOSYS);
   fChain->SetBranchAddress("met_met_NOSYS", &met_met_NOSYS, &b_met_met_NOSYS);
   fChain->SetBranchAddress("met_phi_NOSYS", &met_phi_NOSYS, &b_met_phi_NOSYS);
   fChain->SetBranchAddress("met_significance_NOSYS", &met_significance_NOSYS, &b_met_significance_NOSYS);
   fChain->SetBranchAddress("met_sumet_NOSYS", &met_sumet_NOSYS, &b_met_sumet_NOSYS);
   Notify();
}

bool reco::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. It is normally not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed. The return value is currently not used.

   return true;
}

void reco::Show(Long64_t entry)
{
// Print contents of entry.
// If entry is not specified, print current entry
   if (!fChain) return;
   fChain->Show(entry);
}
Int_t reco::Cut(Long64_t entry)
{
// This function may be called from Loop.
// returns  1 if entry is accepted.
// returns -1 otherwise.
   return 1;
}
#endif // #ifdef reco_cxx
