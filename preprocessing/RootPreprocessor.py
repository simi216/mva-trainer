"""
ROOT file preprocessing module for ttbar dilepton events.

This module replaces the C++ preprocessing pipeline with a pure Python implementation
that handles lepton/jet ordering, feature engineering, and truth matching.
"""

import uproot
import awkward as ak
import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RootPreprocessor:
    """
    Preprocesses ROOT files for ttbar dilepton analysis.
    
    Handles:
    - Event selection (dilepton + jets)
    - Lepton ordering by charge (positive charge first)
    - Jet ordering by pT
    - Truth matching
    - Feature engineering (invariant masses, delta R)
    - NuFlows results extraction
    - Initial state information
    """
    
    def __init__(
        self,
        tree_name: str = "reco",
        save_nu_flows: bool = True,
        save_initial_parton_info: bool = True
    ):
        """
        Initialize preprocessor.
        
        Args:
            tree_name: Name of TTree in ROOT file
            save_nu_flows: Whether to save NuFlows regression targets
            save_initial_parton_info: Whether to save initial parton information
        """
        self.tree_name = tree_name
        self.save_nu_flows = save_nu_flows
        self.save_initial_parton_info = save_initial_parton_info
        
    def process_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        max_events: Optional[int] = None,
        save_format: str = "root"  # "root" or "npz"
    ) -> Dict[str, np.ndarray]:
        """
        Process a ROOT file or directory of ROOT files.
        
        Args:
            input_path: Path to ROOT file or directory
            output_path: Path for output file (optional)
            max_events: Maximum number of events to process
            save_format: Output format ("root" or "npz")
            
        Returns:
            Dictionary of processed arrays
        """
        input_path = Path(input_path)
        
        # Load data
        if input_path.is_dir():
            logger.info(f"Loading from directory: {input_path}")
            data = self._load_from_directory(input_path, max_events)
        else:
            logger.info(f"Loading from file: {input_path}")
            data = self._load_from_file(input_path, max_events)
        
        logger.info(f"Loaded {len(data)} events")
        
        # Apply selection
        logger.info("Applying event selection...")
        mask = self._apply_selection(data)
        logger.info(f"Selected {np.sum(mask)} / {len(mask)} events")
        
        # Filter data
        data_selected = {key: arr[mask] for key, arr in data.items()}
        
        # Process events
        logger.info("Processing events...")
        processed = self._process_events(data_selected)
        
        # Save if output path provided
        if output_path:
            if save_format == "npz":
                self._save_to_npz(processed, output_path)
            elif save_format == "root":
                self._save_to_root(processed, output_path)
            else:
                raise ValueError(f"Unknown save format: {save_format}")
        
        return processed
    
    def _load_from_file(
        self,
        file_path: Path,
        max_events: Optional[int] = None
    ) -> Dict[str, ak.Array]:
        """Load data from a single ROOT file."""
        with uproot.open(file_path) as file:
            tree = file[self.tree_name]
            
            branches = [
                # Electrons
                "el_pt_NOSYS", "el_e_NOSYS", "el_eta", "el_phi", "el_charge",
                # Muons
                "mu_pt_NOSYS", "mu_e_NOSYS", "mu_eta", "mu_phi", "mu_charge",
                # Jets
                "jet_pt_NOSYS", "jet_e_NOSYS", "jet_eta", "jet_phi",
                "jet_GN2v01_Continuous_quantile",
                # Truth matching
                "event_electron_truth_idx", "event_muon_truth_idx",
                "event_jet_truth_idx",
                # Truth information
                "Ttbar_MC_t_beforeFSR_pt", "Ttbar_MC_t_beforeFSR_eta",
                "Ttbar_MC_t_beforeFSR_phi", "Ttbar_MC_t_beforeFSR_m",
                "Ttbar_MC_tbar_beforeFSR_pt", "Ttbar_MC_tbar_beforeFSR_eta",
                "Ttbar_MC_tbar_beforeFSR_phi", "Ttbar_MC_tbar_beforeFSR_m",
                "Ttbar_MC_Wdecay2_afterFSR_from_t_pt",
                "Ttbar_MC_Wdecay2_afterFSR_from_t_eta",
                "Ttbar_MC_Wdecay2_afterFSR_from_t_phi",
                "Ttbar_MC_Wdecay2_afterFSR_from_t_m",
                "Ttbar_MC_Wdecay2_afterFSR_from_tbar_pt",
                "Ttbar_MC_Wdecay2_afterFSR_from_tbar_eta",
                "Ttbar_MC_Wdecay2_afterFSR_from_tbar_phi",
                "Ttbar_MC_Wdecay2_afterFSR_from_tbar_m",
                # Event info
                "eventNumber",
            ]
            
            # Add optional branches
            if self.save_nu_flows:
                branches.append("nuflows_nu_out_NOSYS")
            
            if self.save_initial_parton_info:
                branches.extend(["PDFinfo_PDGID1", "PDFinfo_PDGID2"])
            
            # Load data
            kwargs = {"library": "ak"}
            if max_events:
                kwargs["entry_stop"] = max_events
                
            data = tree.arrays(branches, **kwargs)
            
        return data
    
    def _load_from_directory(
        self,
        directory: Path,
        max_events: Optional[int] = None
    ) -> Dict[str, ak.Array]:
        """Load and concatenate data from all ROOT files in directory."""
        root_files = sorted(directory.glob("*.root"))
        if not root_files:
            raise ValueError(f"No ROOT files found in {directory}")
        
        logger.info(f"Found {len(root_files)} ROOT files")
        
        all_data = []
        total_events = 0
        
        for i, file_path in enumerate(root_files, 1):
            logger.info(f"Loading file {i}/{len(root_files)}: {file_path.name}")
            
            remaining = None
            if max_events:
                remaining = max_events - total_events
                if remaining <= 0:
                    break
            
            data = self._load_from_file(file_path, remaining)
            all_data.append(data)
            total_events += len(data[list(data.keys())[0]])
        
        # Concatenate
        concatenated = {}
        for key in all_data[0].keys():
            concatenated[key] = ak.concatenate([d[key] for d in all_data])
        
        return concatenated
    
    def _apply_selection(self, data: Dict[str, ak.Array]) -> np.ndarray:
        """
        Apply event selection cuts.
        
        Selection:
        - Exactly 2 leptons (e or mu)
        - At least 2 jets
        - Valid truth matching
        - Opposite sign leptons
        """
        n_events = len(data["el_pt_NOSYS"])
        mask = np.ones(n_events, dtype=bool)
        
        # Exactly 2 leptons
        n_leptons = ak.num(data["el_pt_NOSYS"]) + ak.num(data["mu_pt_NOSYS"])
        mask &= (n_leptons == 2)
        
        # At least 2 jets
        n_jets = ak.num(data["jet_pt_NOSYS"])
        mask &= (n_jets >= 2)
        
        # Valid truth matching for jets
        jet_truth_idx = data["event_jet_truth_idx"]
        mask &= (ak.num(jet_truth_idx) >= 6)
        mask &= (jet_truth_idx[:, 0] != -1)
        mask &= (jet_truth_idx[:, 3] != -1)
        mask &= (jet_truth_idx[:, 0] <= 3)
        mask &= (jet_truth_idx[:, 3] <= 3)
        
        # Valid truth matching for leptons
        el_truth_idx = data["event_electron_truth_idx"]
        mu_truth_idx = data["event_muon_truth_idx"]
        
        # At least one matched lepton from top
        has_el0 = (ak.num(el_truth_idx) > 0) & (el_truth_idx[:, 0] != -1)
        has_mu0 = (ak.num(mu_truth_idx) > 0) & (mu_truth_idx[:, 0] != -1)
        mask &= (has_el0 | has_mu0)
        
        # At least one matched lepton from antitop
        has_el1 = (ak.num(el_truth_idx) > 1) & (el_truth_idx[:, 1] != -1)
        has_mu1 = (ak.num(mu_truth_idx) > 1) & (mu_truth_idx[:, 1] != -1)
        mask &= (has_el1 | has_mu1)
        
        # Opposite sign leptons
        el_charge = data["el_charge"]
        mu_charge = data["mu_charge"]
        
        # Case 1: ee
        is_ee = (ak.num(el_charge) == 2) & (ak.num(mu_charge) == 0)
        ee_mask = is_ee & (el_charge[:, 0] != el_charge[:, 1])
        
        # Case 2: mumu
        is_mumu = (ak.num(el_charge) == 0) & (ak.num(mu_charge) == 2)
        mumu_mask = is_mumu & (mu_charge[:, 0] != mu_charge[:, 1])
        
        # Case 3: emu
        is_emu = (ak.num(el_charge) == 1) & (ak.num(mu_charge) == 1)
        emu_mask = is_emu & (el_charge[:, 0] != mu_charge[:, 0])
        
        mask &= (ee_mask | mumu_mask | emu_mask)
        
        # Total charge = 0
        total_charge = ak.sum(el_charge, axis=-1) + ak.sum(mu_charge, axis=-1)
        mask &= (total_charge == 0)
        
        return ak.to_numpy(mask)
    
    def _process_events(self, data: Dict[str, ak.Array]) -> Dict[str, np.ndarray]:
        """Process selected events and create output arrays."""
        n_events = len(data["el_pt_NOSYS"])
        
        # Combine leptons
        leptons = self._combine_leptons(data)
        
        # Order leptons by charge (positive first)
        leptons = self._order_leptons_by_charge(leptons)
        
        # Order jets by pT
        jets = self._order_jets_by_pt(data)
        
        # Compute derived features
        derived = self._compute_derived_features(leptons, jets)
        
        # Extract truth information
        truth = self._extract_truth_info(data, leptons, jets)
        
        # Combine all features
        output = {
            **leptons,
            **jets,
            **derived,
            **truth
        }
        
        # Add optional features
        if self.save_nu_flows:
            output.update(self._extract_nuflows(data))
        
        if self.save_initial_parton_info:
            output.update(self._extract_initial_state(data))
        
        # Add event numbers
        output["eventNumber"] = ak.to_numpy(data["eventNumber"])
        
        return output
    
    def _combine_leptons(self, data: Dict[str, ak.Array]) -> Dict[str, np.ndarray]:
        """Combine electrons and muons into unified lepton arrays."""
        n_events = len(data["el_pt_NOSYS"])
        
        # Get lepton counts
        n_el = ak.num(data["el_pt_NOSYS"])
        n_mu = ak.num(data["mu_pt_NOSYS"])
        
        # Initialize output arrays (max 2 leptons)
        lep_pt = np.zeros((n_events, 2))
        lep_eta = np.zeros((n_events, 2))
        lep_phi = np.zeros((n_events, 2))
        lep_e = np.zeros((n_events, 2))
        lep_charge = np.zeros((n_events, 2))
        lep_pid = np.zeros((n_events, 2))
        lep_truth_idx = np.full((n_events, 2), -1, dtype=int)
        
        # Process each event
        for i in range(n_events):
            leps = []
            
            # Add electrons
            for j in range(n_el[i]):
                lep_dict = {
                    "pt": data["el_pt_NOSYS"][i][j],
                    "eta": data["el_eta"][i][j],
                    "phi": data["el_phi"][i][j],
                    "e": data["el_e_NOSYS"][i][j],
                    "charge": data["el_charge"][i][j],
                    "pid": int(data["el_charge"][i][j]) * 11,
                    "truth_idx": -1
                }
                
                # Check truth matching
                el_truth_idx = data["event_electron_truth_idx"][i]
                if len(el_truth_idx) > 0 and el_truth_idx[0] == j:
                    lep_dict["truth_idx"] = 0  # from top
                elif len(el_truth_idx) > 1 and el_truth_idx[1] == j:
                    lep_dict["truth_idx"] = 1  # from antitop
                
                leps.append(lep_dict)
            
            # Add muons
            for j in range(n_mu[i]):
                lep_dict = {
                    "pt": data["mu_pt_NOSYS"][i][j],
                    "eta": data["mu_eta"][i][j],
                    "phi": data["mu_phi"][i][j],
                    "e": data["mu_e_NOSYS"][i][j],
                    "charge": data["mu_charge"][i][j],
                    "pid": int(data["mu_charge"][i][j]) * 13,
                    "truth_idx": -1
                }
                
                # Check truth matching
                mu_truth_idx = data["event_muon_truth_idx"][i]
                if len(mu_truth_idx) > 0 and mu_truth_idx[0] == j:
                    lep_dict["truth_idx"] = 0  # from top
                elif len(mu_truth_idx) > 1 and mu_truth_idx[1] == j:
                    lep_dict["truth_idx"] = 1  # from antitop
                
                leps.append(lep_dict)
            
            # Store leptons (should be exactly 2 after selection)
            for j, lep in enumerate(leps[:2]):
                lep_pt[i, j] = lep["pt"]
                lep_eta[i, j] = lep["eta"]
                lep_phi[i, j] = lep["phi"]
                lep_e[i, j] = lep["e"]
                lep_charge[i, j] = lep["charge"]
                lep_pid[i, j] = lep["pid"]
                lep_truth_idx[i, j] = lep["truth_idx"]
        
        return {
            "lep_pt": lep_pt,
            "lep_eta": lep_eta,
            "lep_phi": lep_phi,
            "lep_e": lep_e,
            "lep_charge": lep_charge,
            "lep_pid": lep_pid,
            "event_lepton_truth_idx": lep_truth_idx
        }
    
    def _order_leptons_by_charge(
        self,
        leptons: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Order leptons so positive charge comes first."""
        n_events = leptons["lep_charge"].shape[0]
        
        for i in range(n_events):
            # If second lepton has positive charge, swap
            if leptons["lep_charge"][i, 1] > leptons["lep_charge"][i, 0]:
                for key in leptons:
                    leptons[key][i, [0, 1]] = leptons[key][i, [1, 0]]
        
        return leptons
    
    def _order_jets_by_pt(self, data: Dict[str, ak.Array]) -> Dict[str, np.ndarray]:
        """Order jets by pT and extract truth matching."""
        n_events = len(data["jet_pt_NOSYS"])
        max_jets = max(ak.num(data["jet_pt_NOSYS"]))
        
        # Initialize output arrays
        jet_pt = np.full((n_events, max_jets), -999.0)
        jet_eta = np.full((n_events, max_jets), -999.0)
        jet_phi = np.full((n_events, max_jets), -999.0)
        jet_e = np.full((n_events, max_jets), -999.0)
        jet_btag = np.full((n_events, max_jets), -999.0)
        jet_truth_idx = np.full((n_events, 6), -1, dtype=int)
        
        for i in range(n_events):
            n_jets_evt = ak.num(data["jet_pt_NOSYS"][i])
            
            # Create jet list with truth info
            jets_with_truth = []
            for j in range(n_jets_evt):
                truth_label = -1
                evt_jet_truth = data["event_jet_truth_idx"][i]
                
                if len(evt_jet_truth) > 0 and evt_jet_truth[0] == j:
                    truth_label = 0  # b from top
                elif len(evt_jet_truth) > 3 and evt_jet_truth[3] == j:
                    truth_label = 3  # b from antitop
                
                jets_with_truth.append({
                    "pt": data["jet_pt_NOSYS"][i][j],
                    "eta": data["jet_eta"][i][j],
                    "phi": data["jet_phi"][i][j],
                    "e": data["jet_e_NOSYS"][i][j],
                    "btag": data["jet_GN2v01_Continuous_quantile"][i][j],
                    "truth_label": truth_label,
                    "orig_idx": j
                })
            
            # Sort by pT
            jets_with_truth.sort(key=lambda x: x["pt"], reverse=True)
            
            # Fill arrays
            for j, jet in enumerate(jets_with_truth):
                jet_pt[i, j] = jet["pt"]
                jet_eta[i, j] = jet["eta"]
                jet_phi[i, j] = jet["phi"]
                jet_e[i, j] = jet["e"]
                jet_btag[i, j] = jet["btag"]
                
                # Update truth indices based on new ordering
                if jet["truth_label"] == 0:
                    jet_truth_idx[i, 0] = j
                elif jet["truth_label"] == 3:
                    jet_truth_idx[i, 3] = j
        
        return {
            "ordered_jet_pt": jet_pt,
            "ordered_jet_eta": jet_eta,
            "ordered_jet_phi": jet_phi,
            "ordered_jet_e": jet_e,
            "ordered_jet_b_tag": jet_btag,
            "ordered_event_jet_truth_idx": jet_truth_idx,
            "N_jets": ak.to_numpy(ak.num(data["jet_pt_NOSYS"]))
        }
    
    def _compute_derived_features(
        self,
        leptons: Dict[str, np.ndarray],
        jets: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Compute invariant masses and delta R between leptons and jets."""
        n_events = leptons["lep_pt"].shape[0]
        n_jets = jets["ordered_jet_pt"].shape[1]
        
        # Initialize arrays
        m_l1j = np.full((n_events, n_jets), -999.0)
        m_l2j = np.full((n_events, n_jets), -999.0)
        dR_l1j = np.full((n_events, n_jets), -999.0)
        dR_l2j = np.full((n_events, n_jets), -999.0)
        dR_l1l2 = np.zeros(n_events)
        
        for i in range(n_events):
            # Lepton 4-vectors
            px_l1 = leptons["lep_pt"][i, 0] * np.cos(leptons["lep_phi"][i, 0])
            py_l1 = leptons["lep_pt"][i, 0] * np.sin(leptons["lep_phi"][i, 0])
            pz_l1 = leptons["lep_pt"][i, 0] * np.sinh(leptons["lep_eta"][i, 0])
            e_l1 = leptons["lep_e"][i, 0]
            
            px_l2 = leptons["lep_pt"][i, 1] * np.cos(leptons["lep_phi"][i, 1])
            py_l2 = leptons["lep_pt"][i, 1] * np.sin(leptons["lep_phi"][i, 1])
            pz_l2 = leptons["lep_pt"][i, 1] * np.sinh(leptons["lep_eta"][i, 1])
            e_l2 = leptons["lep_e"][i, 1]
            
            # Delta R between leptons
            deta = leptons["lep_eta"][i, 0] - leptons["lep_eta"][i, 1]
            dphi = leptons["lep_phi"][i, 0] - leptons["lep_phi"][i, 1]
            dphi = np.arctan2(np.sin(dphi), np.cos(dphi))  # wrap to [-pi, pi]
            dR_l1l2[i] = np.sqrt(deta**2 + dphi**2)
            
            # Process each jet
            for j in range(n_jets):
                if jets["ordered_jet_pt"][i, j] == -999.0:
                    continue
                
                # Jet 4-vector
                px_j = jets["ordered_jet_pt"][i, j] * np.cos(jets["ordered_jet_phi"][i, j])
                py_j = jets["ordered_jet_pt"][i, j] * np.sin(jets["ordered_jet_phi"][i, j])
                pz_j = jets["ordered_jet_pt"][i, j] * np.sinh(jets["ordered_jet_eta"][i, j])
                e_j = jets["ordered_jet_e"][i, j]
                
                # Invariant mass with l1
                px_l1j = px_l1 + px_j
                py_l1j = py_l1 + py_j
                pz_l1j = pz_l1 + pz_j
                e_l1j = e_l1 + e_j
                m_l1j[i, j] = np.sqrt(e_l1j**2 - px_l1j**2 - py_l1j**2 - pz_l1j**2)
                
                # Invariant mass with l2
                px_l2j = px_l2 + px_j
                py_l2j = py_l2 + py_j
                pz_l2j = pz_l2 + pz_j
                e_l2j = e_l2 + e_j
                m_l2j[i, j] = np.sqrt(e_l2j**2 - px_l2j**2 - py_l2j**2 - pz_l2j**2)
                
                # Delta R with leptons
                deta_l1 = leptons["lep_eta"][i, 0] - jets["ordered_jet_eta"][i, j]
                dphi_l1 = leptons["lep_phi"][i, 0] - jets["ordered_jet_phi"][i, j]
                dphi_l1 = np.arctan2(np.sin(dphi_l1), np.cos(dphi_l1))
                dR_l1j[i, j] = np.sqrt(deta_l1**2 + dphi_l1**2)
                
                deta_l2 = leptons["lep_eta"][i, 1] - jets["ordered_jet_eta"][i, j]
                dphi_l2 = leptons["lep_phi"][i, 1] - jets["ordered_jet_phi"][i, j]
                dphi_l2 = np.arctan2(np.sin(dphi_l2), np.cos(dphi_l2))
                dR_l2j[i, j] = np.sqrt(deta_l2**2 + dphi_l2**2)
        
        return {
            "m_l1j": m_l1j,
            "m_l2j": m_l2j,
            "dR_l1j": dR_l1j,
            "dR_l2j": dR_l2j,
            "dR_l1l2": dR_l1l2
        }
    
    def _extract_truth_info(
        self,
        data: Dict[str, ak.Array],
        leptons: Dict[str, np.ndarray],
        jets: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Extract truth-level information."""
        n_events = len(data["Ttbar_MC_t_beforeFSR_pt"])
        
        # Top and antitop 4-vectors
        px_t = ak.to_numpy(data["Ttbar_MC_t_beforeFSR_pt"] * np.cos(data["Ttbar_MC_t_beforeFSR_phi"]))
        py_t = ak.to_numpy(data["Ttbar_MC_t_beforeFSR_pt"] * np.sin(data["Ttbar_MC_t_beforeFSR_phi"]))
        pz_t = ak.to_numpy(data["Ttbar_MC_t_beforeFSR_pt"] * np.sinh(data["Ttbar_MC_t_beforeFSR_eta"]))
        m_t = ak.to_numpy(data["Ttbar_MC_t_beforeFSR_m"])
        e_t = np.sqrt(px_t**2 + py_t**2 + pz_t**2 + m_t**2)
        
        px_tbar = ak.to_numpy(data["Ttbar_MC_tbar_beforeFSR_pt"] * np.cos(data["Ttbar_MC_tbar_beforeFSR_phi"]))
        py_tbar = ak.to_numpy(data["Ttbar_MC_tbar_beforeFSR_pt"] * np.sin(data["Ttbar_MC_tbar_beforeFSR_phi"]))
        pz_tbar = ak.to_numpy(data["Ttbar_MC_tbar_beforeFSR_pt"] * np.sinh(data["Ttbar_MC_tbar_beforeFSR_eta"]))
        m_tbar = ak.to_numpy(data["Ttbar_MC_tbar_beforeFSR_m"])
        e_tbar = np.sqrt(px_tbar**2 + py_tbar**2 + pz_tbar**2 + m_tbar**2)
        
        # ttbar system
        px_tt = px_t + px_tbar
        py_tt = py_t + py_tbar
        pz_tt = pz_t + pz_tbar
        e_tt = e_t + e_tbar
        p_tt = np.sqrt(px_tt**2 + py_tt**2 + pz_tt**2)
        
        truth_ttbar_mass = np.sqrt(e_tt**2 - p_tt**2)
        truth_ttbar_pt = np.sqrt(px_tt**2 + py_tt**2)
        truth_tt_boost_parameter = p_tt / e_tt
        
        # Neutrino information (from W decay)
        px_nu_t = ak.to_numpy(data["Ttbar_MC_Wdecay2_afterFSR_from_t_pt"] * 
                              np.cos(data["Ttbar_MC_Wdecay2_afterFSR_from_t_phi"]))
        py_nu_t = ak.to_numpy(data["Ttbar_MC_Wdecay2_afterFSR_from_t_pt"] * 
                              np.sin(data["Ttbar_MC_Wdecay2_afterFSR_from_t_phi"]))
        pz_nu_t = ak.to_numpy(data["Ttbar_MC_Wdecay2_afterFSR_from_t_pt"] * 
                              np.sinh(data["Ttbar_MC_Wdecay2_afterFSR_from_t_eta"]))
        m_nu_t = ak.to_numpy(data["Ttbar_MC_Wdecay2_afterFSR_from_t_m"])
        e_nu_t = np.sqrt(px_nu_t**2 + py_nu_t**2 + pz_nu_t**2 + m_nu_t**2)
        
        px_nu_tbar = ak.to_numpy(data["Ttbar_MC_Wdecay2_afterFSR_from_tbar_pt"] * 
                                 np.cos(data["Ttbar_MC_Wdecay2_afterFSR_from_tbar_phi"]))
        py_nu_tbar = ak.to_numpy(data["Ttbar_MC_Wdecay2_afterFSR_from_tbar_pt"] * 
                                 np.sin(data["Ttbar_MC_Wdecay2_afterFSR_from_tbar_phi"]))
        pz_nu_tbar = ak.to_numpy(data["Ttbar_MC_Wdecay2_afterFSR_from_tbar_pt"] * 
                                 np.sinh(data["Ttbar_MC_Wdecay2_afterFSR_from_tbar_eta"]))
        m_nu_tbar = ak.to_numpy(data["Ttbar_MC_Wdecay2_afterFSR_from_tbar_m"])
        e_nu_tbar = np.sqrt(px_nu_tbar**2 + py_nu_tbar**2 + pz_nu_tbar**2 + m_nu_tbar**2)
        
        return {
            "truth_ttbar_mass": truth_ttbar_mass,
            "truth_ttbar_pt": truth_ttbar_pt,
            "truth_tt_boost_parameter": truth_tt_boost_parameter,
            "truth_nu_px": np.stack([px_nu_t, px_nu_tbar], axis=1),
            "truth_nu_py": np.stack([py_nu_t, py_nu_tbar], axis=1),
            "truth_nu_pz": np.stack([pz_nu_t, pz_nu_tbar], axis=1),
            "truth_nu_e": np.stack([e_nu_t, e_nu_tbar], axis=1),
        }
    
    def _extract_nuflows(self, data: Dict[str, ak.Array]) -> Dict[str, np.ndarray]:
        """Extract NuFlows regression results."""
        nuflows = data["nuflows_nu_out_NOSYS"]
        
        # NuFlows output is [nu1_px, nu1_py, nu1_pz, nu1_e, nu2_px, nu2_py, nu2_pz, nu2_e]
        nuflows_np = ak.to_numpy(nuflows)
        
        return {
            "nuflows_nu_px": nuflows_np[:, [0, 4]],
            "nuflows_nu_py": nuflows_np[:, [1, 5]],
            "nuflows_nu_pz": nuflows_np[:, [2, 6]],
            "nuflows_nu_e": nuflows_np[:, [3, 7]],
        }
    
    def _extract_initial_state(self, data: Dict[str, ak.Array]) -> Dict[str, np.ndarray]:
        """Extract initial state parton information."""
        return {
            "PDFinfo_PDGID1": ak.to_numpy(data["PDFinfo_PDGID1"]),
            "PDFinfo_PDGID2": ak.to_numpy(data["PDFinfo_PDGID2"]),
        }
    
    def _save_to_npz(self, data: Dict[str, np.ndarray], output_path: str):
        """Save processed data to NPZ format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving to NPZ: {output_path}")
        np.savez_compressed(output_path, **data)
        logger.info(f"Saved {len(data)} arrays to {output_path}")
    
    def _save_to_root(self, data: Dict[str, np.ndarray], output_path: str):
        """Save processed data to ROOT format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving to ROOT: {output_path}")
        
        with uproot.recreate(output_path) as file:
            # Convert numpy arrays to awkward arrays for ROOT
            awkward_data = {}
            for key, arr in data.items():
                if arr.ndim == 1:
                    awkward_data[key] = ak.Array(arr)
                else:
                    # Multi-dimensional arrays need special handling
                    awkward_data[key] = ak.Array(arr)
            
            file["reco"] = awkward_data
        
        logger.info(f"Saved {len(data)} branches to {output_path}")


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Preprocess ROOT files for ttbar dilepton analysis"
    )
    parser.add_argument("input", help="Input ROOT file or directory")
    parser.add_argument("output", help="Output file path")
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Maximum number of events to process"
    )
    parser.add_argument(
        "--format",
        choices=["root", "npz"],
        default="npz",
        help="Output format (default: npz)"
    )
    parser.add_argument(
        "--no-nuflows",
        action="store_true",
        help="Don't save NuFlows results"
    )
    parser.add_argument(
        "--no-initial-state",
        action="store_true",
        help="Don't save initial parton information"
    )
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = RootPreprocessor(
        save_nu_flows=not args.no_nuflows,
        save_initial_parton_info=not args.no_initial_state
    )
    
    # Process file
    preprocessor.process_file(
        input_path=args.input,
        output_path=args.output,
        max_events=args.max_events,
        save_format=args.format
    )
    
    logger.info("Done!")


if __name__ == "__main__":
    main()