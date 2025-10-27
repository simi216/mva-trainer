#include "../include/PreProcessor.h"

PreProcessor::PreProcessor(const char *inFileName, const char *outFileName, const char *treeName)
{
    TreeName = treeName;
    inputFileName = std::string(inFileName);
    outputFileName = outFileName;
    inputFile = nullptr;

    if (std::filesystem::is_directory(inFileName))
    {
        inputTree = new TChain(treeName);
        for (const auto &entry : std::filesystem::directory_iterator(inFileName))
        {
            if (entry.path().extension() == ".root")
            {
                std::string fileName = entry.path().string();
                inputTree->Add(fileName.c_str());
                if (!inputTree)
                {
                    std::cerr << "Error getting tree from input file: " << fileName << std::endl;
                    return;
                }
            }
        }
    }
    else
    {
        inputFile = new TFile(inputFileName.c_str(), "read");
        if (!inputFile || inputFile->IsZombie())
        {
            std::cerr << "Error opening input file: " << inputFileName << std::endl;
            if (inputFile)
                delete inputFile;
            inputFile = nullptr;
            return;
        }
        inputTree = new TChain(treeName);
        inputTree->Add(inputFileName.c_str());
        if (!inputTree)
        {
            std::cerr << "Error getting tree from input file: " << inputFileName << std::endl;
            if (outputFile)
            {
                outputFile->Close();
                delete outputFile;
            }
            if (inputFile)
            {
                inputFile->Close();
                delete inputFile;
            }
            return;
        }
    }

    outputFile = new TFile(outputFileName.c_str(), "recreate");
    if (!outputFile || outputFile->IsZombie())
    {
        std::cerr << "Error creating output file: " << outputFileName << std::endl;
        if (outputFile)
            delete outputFile;
        outputFile = nullptr;
        if (inputFile)
        {
            inputFile->Close();
            delete inputFile;
        }
        return;
    }

    outputFile->cd();
    outputTree = inputTree->CloneTree(0);
    if (!outputTree)
    {
        std::cerr << "Error cloning tree from input file: " << inputFileName << std::endl;
        return;
    }
    outputTree->SetName("reco");
    outputTree->SetDirectory(outputFile);
}

PreProcessor::~PreProcessor()
{
    if (inputFile)
    {
        inputFile->Close();
        delete inputFile;
    }
    if (outputFile)
    {
        outputFile->Close();
        delete outputFile;
    }
}

void PreProcessor::RegisterBranches()
{
    outputTree->Branch("lep_pt", &lep_pt);
    outputTree->Branch("lep_eta", &lep_eta);
    outputTree->Branch("lep_phi", &lep_phi);
    outputTree->Branch("lep_e", &lep_e);
    outputTree->Branch("lep_charge", &lep_charge);
    outputTree->Branch("lep_pid", &lep_pid);
    outputTree->Branch("event_lepton_truth_idx", &event_lepton_truth_idx);

    outputTree->Branch("ordered_jet_pt", &jet_pt);
    outputTree->Branch("ordered_jet_eta", &jet_eta);
    outputTree->Branch("ordered_jet_phi", &jet_phi);
    outputTree->Branch("ordered_jet_e", &jet_e);
    outputTree->Branch("ordered_jet_b_tag", &jet_btag);
    outputTree->Branch("ordered_event_jet_truth_idx", &event_jet_truth_idx);

    outputTree->Branch("m_l1j", &m_l1j);
    outputTree->Branch("m_l2j", &m_l2j);
    outputTree->Branch("dR_l1j", &dR_l1j);
    outputTree->Branch("dR_l2j", &dR_l2j);

    outputTree->Branch("truth_ttbar_mass", &truth_ttbar_mass);
    outputTree->Branch("truth_ttbar_pt", &truth_ttbar_pt);
    outputTree->Branch("N_jets", &N_jets);
    outputTree->Branch("dR_l1l2", &dR_l1l2);
    outputTree->Branch("truth_tt_boost_parameter", &truth_tt_boost_parameter);

    outputTree->Branch("truth_top_mass", &truth_top_mass);
    outputTree->Branch("truth_top_pt", &truth_top_pt);
    outputTree->Branch("truth_top_eta", &truth_top_eta);
    outputTree->Branch("truth_top_phi", &truth_top_phi);
    outputTree->Branch("truth_top_e", &truth_top_e);

    outputTree->Branch("truth_tbar_mass", &truth_tbar_mass);
    outputTree->Branch("truth_tbar_pt", &truth_tbar_pt);
    outputTree->Branch("truth_tbar_eta", &truth_tbar_eta);
    outputTree->Branch("truth_tbar_phi", &truth_tbar_phi);
    outputTree->Branch("truth_tbar_e", &truth_tbar_e);

    outputTree->Branch("truth_top_neutino_mass", &truth_top_neutino_mass);
    outputTree->Branch("truth_top_neutino_pt", &truth_top_neutino_pt);
    outputTree->Branch("truth_top_neutino_eta", &truth_top_neutino_eta);
    outputTree->Branch("truth_top_neutino_phi", &truth_top_neutino_phi);
    outputTree->Branch("truth_top_neutino_e", &truth_top_neutino_e);
    outputTree->Branch("truth_top_neutrino_px", &truth_top_neutrino_px);
    outputTree->Branch("truth_top_neutrino_py", &truth_top_neutrino_py);
    outputTree->Branch("truth_top_neutrino_pz", &truth_top_neutrino_pz);

    outputTree->Branch("truth_tbar_neutino_mass", &truth_tbar_neutino_mass);
    outputTree->Branch("truth_tbar_neutino_pt", &truth_tbar_neutino_pt);
    outputTree->Branch("truth_tbar_neutino_eta", &truth_tbar_neutino_eta);
    outputTree->Branch("truth_tbar_neutino_phi", &truth_tbar_neutino_phi);
    outputTree->Branch("truth_tbar_neutino_e", &truth_tbar_neutino_e);
    outputTree->Branch("truth_tbar_neutrino_px", &truth_tbar_neutrino_px);
    outputTree->Branch("truth_tbar_neutrino_py", &truth_tbar_neutrino_py);
    outputTree->Branch("truth_tbar_neutrino_pz", &truth_tbar_neutrino_pz);

    Event = new EventType(inputTree);
}

void PreProcessor::RegisterNuFlowResults()
{
    outputTree->Branch("nu_flows_neutriono_p_x", &nu_flows_neutriono_p_x);
    outputTree->Branch("nu_flows_neutriono_p_y", &nu_flows_neutriono_p_y);
    outputTree->Branch("nu_flows_neutriono_p_z", &nu_flows_neutriono_p_z);

    outputTree->Branch("nu_flows_anti_neutriono_p_x", &nu_flows_anti_neutriono_p_x);
    outputTree->Branch("nu_flows_anti_neutriono_p_y", &nu_flows_anti_neutriono_p_y);
    outputTree->Branch("nu_flows_anti_neutriono_p_z", &nu_flows_anti_neutriono_p_z);

    outputTree->Branch("nu_flows_m_ttbar", &nu_flows_m_ttbar);

    save_nu_flows = true;
}

void PreProcessor::RegisterInitialStateInfo()
{
    outputTree->Branch("truth_initial_parton_num_gluons", &truth_initial_parton_num_gluons);

    save_initial_parton_info = true;
}

void PreProcessor::GetNuFlowBranches()
{
    nu_flows_neutriono_p_x = Event->nuflows_nu_out_NOSYS->at(0) * 1e3;
    nu_flows_neutriono_p_y = Event->nuflows_nu_out_NOSYS->at(1) * 1e3;
    nu_flows_neutriono_p_z = Event->nuflows_nu_out_NOSYS->at(2) * 1e3;
    nu_flows_anti_neutriono_p_x = Event->nuflows_nu_out_NOSYS->at(3) * 1e3;
    nu_flows_anti_neutriono_p_y = Event->nuflows_nu_out_NOSYS->at(4) * 1e3;
    nu_flows_anti_neutriono_p_z = Event->nuflows_nu_out_NOSYS->at(5) * 1e3;
}

void PreProcessor::SaveNuFlowTTbarMass(TLorentzVector &l1, TLorentzVector &l2, TLorentzVector &j1, TLorentzVector &j2)
{
    TLorentzVector nu1, nu2;
    double E1, E2;
    E1 = std::sqrt(nu_flows_neutriono_p_x * nu_flows_neutriono_p_x + nu_flows_neutriono_p_y * nu_flows_neutriono_p_y + nu_flows_neutriono_p_z * nu_flows_neutriono_p_z);
    E2 = std::sqrt(nu_flows_anti_neutriono_p_x * nu_flows_anti_neutriono_p_x + nu_flows_anti_neutriono_p_y * nu_flows_anti_neutriono_p_y + nu_flows_anti_neutriono_p_z * nu_flows_anti_neutriono_p_z);
    nu1.SetPxPyPzE(nu_flows_neutriono_p_x, nu_flows_neutriono_p_y, nu_flows_neutriono_p_z, E1);
    nu2.SetPxPyPzE(nu_flows_anti_neutriono_p_x, nu_flows_anti_neutriono_p_y, nu_flows_anti_neutriono_p_z, E2);
    TLorentzVector ttbar = l1 + l2 + j1 + j2 + nu1 + nu2;
    nu_flows_m_ttbar = ttbar.M();
    //std::cout << E1<< "  " << E2 << "  " << nu1.M() << "  " << nu2.M() << std::endl;
}

void PreProcessor::FillInitialStateBranches()
{
    int truth_initial_state_1_pdgId = Event->PDFinfo_PDGID1;
    int truth_initial_state_2_pdgId = Event->PDFinfo_PDGID2;

    truth_initial_parton_num_gluons = 0;
    if (truth_initial_state_1_pdgId == 21)
    {
        truth_initial_parton_num_gluons++;
    }
    if (truth_initial_state_2_pdgId == 21)
    {
        truth_initial_parton_num_gluons++;
    }
}

bool PreProcessor::PreSelection()
{
    if ((Event->el_e_NOSYS->size() + Event->mu_e_NOSYS->size()) != 2)
    {
        return false;
    }
    if ((Event->jet_e_NOSYS->size()) < 2)
    {
        return false;
    }
    if (Event->event_jet_truth_idx->size() < 6)
    {
        return false;
    }
    if (Event->event_jet_truth_idx->at(0) == -1 or Event->event_jet_truth_idx->at(3) == -1)
    {
        return false;
    }
    if (Event->event_jet_truth_idx->at(0) > 3 or Event->event_jet_truth_idx->at(3) > 3)
    {
        return false;
    }
    if (Event->event_electron_truth_idx->at(0) == -1 and Event->event_muon_truth_idx->at(0) == -1)
    {
        return false;
    }
    if (Event->event_electron_truth_idx->at(1) == -1 and Event->event_muon_truth_idx->at(1) == -1)
    {
        return false;
    }
    if (Event->jet_e_NOSYS->size() < 2)
    {
        return false;
    }
    if (Event->el_e_NOSYS->size() == 2)
    {
        if (Event->el_charge->at(0) == Event->el_charge->at(1))
        {
            return false;
        }
    }
    else if (Event->mu_e_NOSYS->size() == 2)
    {
        if (Event->mu_charge->at(0) == Event->mu_charge->at(1))
        {
            return false;
        }
    }
    else
    {
        if (Event->el_e_NOSYS->size() == 1 && Event->mu_e_NOSYS->size() == 1)
        {
            if (Event->el_charge->at(0) == Event->mu_charge->at(0))
            {
                return false;
            }
        }
    }
    int charge_sum = 0;
    for (size_t i = 0; i < Event->el_charge->size(); ++i)
    {
        charge_sum += Event->el_charge->at(i);
    }
    for (size_t i = 0; i < Event->mu_charge->size(); ++i)
    {
        charge_sum += Event->mu_charge->at(i);
    }
    if (charge_sum != 0)
    {
        return false;
    }

    return true;
}

void PreProcessor::FillBranches()
{
    std::vector<std::tuple<TLorentzVector, int, int, int>> leptons;
    std::vector<std::tuple<TLorentzVector, int, int>> jets;
    for (size_t i = 0; i < Event->el_e_NOSYS->size(); ++i)
    {
        TLorentzVector lep;
        lep.SetPtEtaPhiE(Event->el_pt_NOSYS->at(i), Event->el_eta->at(i), Event->el_phi->at(i), Event->el_e_NOSYS->at(i));
        if (Event->event_electron_truth_idx->at(0) == static_cast<int>(i))
        {
            leptons.emplace_back(std::make_tuple(lep, Event->el_charge->at(i), static_cast<int>(Event->el_charge->at(i)) * 11, 1));
        }
        else if (Event->event_electron_truth_idx->at(1) == static_cast<int>(i))
        {
            leptons.emplace_back(std::make_tuple(lep, Event->el_charge->at(i), static_cast<int>(Event->el_charge->at(i)) * 11, -1));
        }
        else
        {
            leptons.emplace_back(std::make_tuple(lep, Event->el_charge->at(i), static_cast<int>(Event->el_charge->at(i)) * 11, 0));
        }
    }
    for (size_t i = 0; i < Event->mu_e_NOSYS->size(); ++i)
    {
        TLorentzVector lep;
        lep.SetPtEtaPhiE(Event->mu_pt_NOSYS->at(i), Event->mu_eta->at(i), Event->mu_phi->at(i), Event->mu_e_NOSYS->at(i));
        if (Event->event_muon_truth_idx->at(0) == static_cast<int>(i))
        {
            leptons.emplace_back(std::make_tuple(lep, Event->mu_charge->at(i), static_cast<int>(Event->mu_charge->at(i)) * 13, 1));
        }
        else if (Event->event_muon_truth_idx->at(1) == static_cast<int>(i))
        {
            leptons.emplace_back(std::make_tuple(lep, Event->mu_charge->at(i), static_cast<int>(Event->mu_charge->at(i)) * 13, -1));
        }
        else
        {
            leptons.emplace_back(std::make_tuple(lep, Event->mu_charge->at(i), static_cast<int>(Event->mu_charge->at(i)) * 13, 0));
        }
    }
    for (size_t i = 0; i < Event->jet_pt_NOSYS->size(); ++i)
    {
        TLorentzVector jet;
        jet.SetPtEtaPhiE(Event->jet_pt_NOSYS->at(i), Event->jet_eta->at(i), Event->jet_phi->at(i), Event->jet_e_NOSYS->at(i));
        if (Event->event_jet_truth_idx->at(0) == static_cast<int>(i))
        {
            jets.emplace_back(std::make_tuple(jet, Event->jet_GN2v01_Continuous_quantile->at(i), 1)); // No charge or pid for jets
        }
        else if (Event->event_jet_truth_idx->at(3) == static_cast<int>(i))
        {
            jets.emplace_back(std::make_tuple(jet, Event->jet_GN2v01_Continuous_quantile->at(i), -1)); // No charge or pid for jets
        }
        else
        {
            jets.emplace_back(std::make_tuple(jet, Event->jet_GN2v01_Continuous_quantile->at(i), 0)); // No charge or pid for jets
        }
    }
    std::sort(leptons.begin(), leptons.end(), [](const std::tuple<TLorentzVector, int, int, int> &a, const std::tuple<TLorentzVector, int, int, int> &b)
              { return std::get<1>(a) > std::get<1>(b); });
    std::sort(jets.begin(), jets.end(), [](const auto &a, const auto &b)
              { return std::get<0>(a).Pt() > std::get<0>(b).Pt(); });
    lep_pt.clear();
    lep_eta.clear();
    lep_phi.clear();
    lep_e.clear();
    lep_charge.clear();
    lep_pid.clear();
    event_lepton_truth_idx.clear();
    for (size_t i = 0; i < 2; ++i)
    {
        event_lepton_truth_idx.push_back(-1);
    }
    for (size_t i = 0; i < leptons.size(); ++i)
    {
        lep_pt.push_back(std::get<0>(leptons[i]).Pt());
        lep_eta.push_back(std::get<0>(leptons[i]).Eta());
        lep_phi.push_back(std::get<0>(leptons[i]).Phi());
        lep_e.push_back(std::get<0>(leptons[i]).E());
        lep_charge.push_back(std::get<1>(leptons[i]));
        lep_pid.push_back(std::get<2>(leptons[i]));
        if (std::get<3>(leptons[i]) == 1)
        {
            event_lepton_truth_idx.at(0) = i;
        }
        else if (std::get<3>(leptons[i]) == -1)
        {
            event_lepton_truth_idx.at(1) = i;
        }
    }
    jet_pt.clear();
    jet_eta.clear();
    jet_phi.clear();
    jet_e.clear();
    jet_btag.clear();
    event_jet_truth_idx.clear();
    for (int i = 0; i < 6; ++i)
    {
        event_jet_truth_idx.push_back(-1);
    }

    for (size_t i = 0; i < jets.size(); ++i)
    {
        jet_pt.push_back(std::get<0>(jets[i]).Pt());
        jet_eta.push_back(std::get<0>(jets[i]).Eta());
        jet_phi.push_back(std::get<0>(jets[i]).Phi());
        jet_e.push_back(std::get<0>(jets[i]).E());
        jet_btag.push_back(std::get<1>(jets[i]));
        if (std::get<2>(jets[i]) == 1)
        {
            event_jet_truth_idx.at(0) = i;
        }
        else if (std::get<2>(jets[i]) == -1)
        {
            event_jet_truth_idx.at(3) = i;
        }
    }

    m_l1j.clear();
    m_l2j.clear();
    dR_l1j.clear();
    dR_l2j.clear();
    for (size_t i = 0; i < jets.size(); ++i)
    {
        TLorentzVector l1j = std::get<0>(leptons[0]) + std::get<0>(jets[i]);
        m_l1j.push_back(l1j.M());
        dR_l1j.push_back(std::get<0>(leptons[0]).DeltaR(std::get<0>(jets[i])));
        TLorentzVector l2j = std::get<0>(leptons[1]) + std::get<0>(jets[i]);
        m_l2j.push_back(l2j.M());
        dR_l2j.push_back(std::get<0>(leptons[1]).DeltaR(std::get<0>(jets[i])));
    }
    TLorentzVector top;
    top.SetPtEtaPhiM(Event->Ttbar_MC_t_beforeFSR_pt, Event->Ttbar_MC_t_beforeFSR_eta, Event->Ttbar_MC_t_beforeFSR_phi, Event->Ttbar_MC_t_beforeFSR_m);
    TLorentzVector topbar;
    topbar.SetPtEtaPhiM(Event->Ttbar_MC_tbar_beforeFSR_pt, Event->Ttbar_MC_tbar_beforeFSR_eta, Event->Ttbar_MC_tbar_beforeFSR_phi, Event->Ttbar_MC_tbar_beforeFSR_m);
    truth_ttbar_mass = (top + topbar).M();
    truth_ttbar_pt = (top + topbar).Pt();
    truth_tt_boost_parameter = (top + topbar).P() / (top + topbar).E();
    dR_l1l2 = std::get<0>(leptons[0]).DeltaR(std::get<0>(leptons[1]));

    TLorentzVector neutrino_top;
    TLorentzVector neutrino_tbar;

    neutrino_top.SetPtEtaPhiM(Event->Ttbar_MC_Wdecay2_afterFSR_from_t_pt, Event->Ttbar_MC_Wdecay2_afterFSR_from_t_eta, Event->Ttbar_MC_Wdecay2_afterFSR_from_t_phi, Event->Ttbar_MC_Wdecay2_afterFSR_from_t_m);
    neutrino_tbar.SetPtEtaPhiM(Event->Ttbar_MC_Wdecay2_afterFSR_from_tbar_pt, Event->Ttbar_MC_Wdecay2_afterFSR_from_tbar_eta, Event->Ttbar_MC_Wdecay2_afterFSR_from_tbar_phi, Event->Ttbar_MC_Wdecay2_afterFSR_from_tbar_m);

    truth_top_neutino_mass = neutrino_top.M();
    truth_top_neutino_pt = neutrino_top.Pt();
    truth_top_neutino_eta = neutrino_top.Eta();
    truth_top_neutino_phi = neutrino_top.Phi();
    truth_top_neutino_e = neutrino_top.E();
    truth_tbar_neutino_mass = neutrino_tbar.M();
    truth_tbar_neutino_pt = neutrino_tbar.Pt();
    truth_tbar_neutino_eta = neutrino_tbar.Eta();
    truth_tbar_neutino_phi = neutrino_tbar.Phi();
    truth_tbar_neutino_e = neutrino_tbar.E();
    truth_top_neutrino_px = neutrino_top.Px();
    truth_top_neutrino_py = neutrino_top.Py();
    truth_top_neutrino_pz = neutrino_top.Pz();
    truth_tbar_neutrino_px = neutrino_tbar.Px();
    truth_tbar_neutrino_py = neutrino_tbar.Py();
    truth_tbar_neutrino_pz = neutrino_tbar.Pz();

    N_jets = Event->jet_e_NOSYS->size();
    truth_top_mass = Event->Ttbar_MC_t_beforeFSR_m;
    truth_top_pt = Event->Ttbar_MC_t_beforeFSR_pt;
    truth_top_eta = Event->Ttbar_MC_t_beforeFSR_eta;
    truth_top_phi = Event->Ttbar_MC_t_beforeFSR_phi;
    truth_top_e = top.E();
    truth_tbar_mass = Event->Ttbar_MC_tbar_beforeFSR_m;
    truth_tbar_pt = Event->Ttbar_MC_tbar_beforeFSR_pt;
    truth_tbar_eta = Event->Ttbar_MC_tbar_beforeFSR_eta;
    truth_tbar_phi = Event->Ttbar_MC_tbar_beforeFSR_phi;
    truth_tbar_e = topbar.E();

    if (save_nu_flows)
    {
        GetNuFlowBranches();
        int true_jet_index_1 = Event->event_jet_truth_idx->at(0);
        int true_jet_index_2 = Event->event_jet_truth_idx->at(3);
        SaveNuFlowTTbarMass(std::get<0>(leptons[0]), std::get<0>(leptons[1]), std::get<0>(jets[true_jet_index_1]), std::get<0>(jets[true_jet_index_2]));
    }
    if (save_initial_parton_info)
    {
        FillInitialStateBranches();
    }

    outputTree->Fill();
}

void PreProcessor::Process()
{
    RegisterBranches();
    if (!inputTree)
    {
        std::cerr << "Error: No input tree found." << std::endl;
        return;
    }
    Long64_t nEntries = inputTree->GetEntries();
    for (Long64_t i = 0; i < nEntries; ++i)
    {
        inputTree->GetEntry(i);
        if (i % 1000 == 0)
        {
            std::cout << "Processing entry " << i << " / " << nEntries << std::endl;
        }
        if (PreSelection())
        {
            FillBranches();
        }
    }
    outputFile->cd();
    outputTree->SetDirectory(nullptr);
    outputTree->Write("", TObject::kOverwrite);
    outputFile->Close();
}