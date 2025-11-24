#!/usr/bin/env python3
"""
Test integration between RootPreprocessor and DataLoader.

This script verifies that DataLoader can properly read NPZ files
produced by RootPreprocessor using LoadConfig to specify which keys to load.
"""

import sys
import os
import tempfile
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.RootPreprocessor import RootPreprocessor
from core.DataLoader import DataPreprocessor
from core.Configs import LoadConfig


def create_mock_preprocessed_npz(path: str):
    """Create a mock NPZ file with RootPreprocessor format."""
    n_events = 100
    max_jets = 10
    
    # Create flat structure as produced by RootPreprocessor
    data = {
        # Leptons (always 2 after preprocessing)
        'lep_pt': np.random.uniform(20, 200, (n_events, 2)),
        'lep_eta': np.random.uniform(-2.5, 2.5, (n_events, 2)),
        'lep_phi': np.random.uniform(-np.pi, np.pi, (n_events, 2)),
        'lep_e': np.random.uniform(25, 210, (n_events, 2)),
        'lep_charge': np.random.choice([-1, 1], (n_events, 2)),
        'lep_pid': np.random.choice([11, 13], (n_events, 2)),
        
        # Jets (variable number per event, padded to max_jets)
        'ordered_jet_pt': np.random.uniform(20, 300, (n_events, max_jets)),
        'ordered_jet_eta': np.random.uniform(-2.5, 2.5, (n_events, max_jets)),
        'ordered_jet_phi': np.random.uniform(-np.pi, np.pi, (n_events, max_jets)),
        'ordered_jet_e': np.random.uniform(25, 310, (n_events, max_jets)),
        'ordered_jet_b_tag': np.random.uniform(0, 1, (n_events, max_jets)),
        
        # Truth indices for assignment
        'event_lepton_truth_idx': np.random.randint(-1, 2, (n_events, 2)),
        'ordered_event_jet_truth_idx': np.random.randint(-1, 6, (n_events, 6)),
        
        # Truth info for regression
        'truth_top_neutrino_px': np.random.uniform(-100, 100, n_events),
        'truth_top_neutrino_py': np.random.uniform(-100, 100, n_events),
        'truth_top_neutrino_pz': np.random.uniform(-200, 200, n_events),
        'truth_tbar_neutrino_px': np.random.uniform(-100, 100, n_events),
        'truth_tbar_neutrino_py': np.random.uniform(-100, 100, n_events),
        'truth_tbar_neutrino_pz': np.random.uniform(-200, 200, n_events),
        
        # Additional truth info
        'truth_ttbar_mass': np.random.uniform(300, 700, n_events),
        'truth_top_mass': np.random.uniform(170, 175, n_events),
        
        # Event metadata
        'event_weight': np.random.uniform(0.5, 1.5, n_events),
        'mc_event_number': np.arange(n_events),
        'N_jets': np.random.randint(6, max_jets + 1, n_events),
    }
    
    np.savez_compressed(path, **data)
    print(f"Created mock NPZ file: {path}")


def test_dataloader_flat_format():
    """Test that DataLoader can read flat format from RootPreprocessor with LoadConfig."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = os.path.join(tmpdir, "test_data.npz")
        
        # Create mock data
        create_mock_preprocessed_npz(npz_path)
        
        # Create LoadConfig specifying what to load
        load_config = LoadConfig(
            jet_features=['pt', 'eta', 'phi', 'e', 'b'],
            lepton_features=['pt', 'eta', 'phi', 'e'],
            met_features=[],
            non_training_features=['truth_ttbar_mass', 'truth_top_mass'],
            jet_truth_label='ordered_event_jet_truth_idx',
            lepton_truth_label='event_lepton_truth_idx',
            max_jets=10,
            NUM_LEPTONS=2,
            event_weight='event_weight',
            mc_event_number='mc_event_number',
            neutrino_momentum_features=['px', 'py', 'pz'],
            antineutrino_momentum_features=['px', 'py', 'pz'],
        )
        
        # Load with DataLoader
        print("\nLoading data with DataPreprocessor...")
        preprocessor = DataPreprocessor(load_config)
        data_config = preprocessor.load_from_npz(npz_path)
        
        print(f"\nData config: {data_config}")
        print(f"Data length: {preprocessor.data_length}")
        
        # Verify structure
        print("\nVerifying data structure...")
        assert preprocessor.feature_data is not None, "Feature data not loaded"
        
        # Check expected keys
        expected_keys = ['lepton', 'jet']
        for key in expected_keys:
            assert key in preprocessor.feature_data, f"Missing key: {key}"
            print(f"✓ {key}: shape {preprocessor.feature_data[key].shape}")
        
        # Verify shapes
        n_events = preprocessor.data_length
        assert preprocessor.feature_data['lepton'].shape == (n_events, 2, 4), \
            f"Wrong lepton shape: {preprocessor.feature_data['lepton'].shape}"
        
        max_jets = preprocessor.feature_data['jet'].shape[1]
        n_jet_features = preprocessor.feature_data['jet'].shape[2]
        assert preprocessor.feature_data['jet'].shape == (n_events, max_jets, n_jet_features), \
            f"Wrong jet shape: {preprocessor.feature_data['jet'].shape}"
        
        # Check optional keys
        if 'assignment_labels' in preprocessor.feature_data:
            print(f"✓ assignment_labels: shape {preprocessor.feature_data['assignment_labels'].shape}")
            assert preprocessor.feature_data['assignment_labels'].shape == (n_events, max_jets, 2)
        
        if 'neutrino_truth' in preprocessor.feature_data:
            print(f"✓ neutrino_truth: shape {preprocessor.feature_data['neutrino_truth'].shape}")
            assert preprocessor.feature_data['neutrino_truth'].shape == (n_events, 2, 3)
        
        if 'event_weight' in preprocessor.feature_data:
            print(f"✓ event_weight: shape {preprocessor.feature_data['event_weight'].shape}")
            assert len(preprocessor.feature_data['event_weight']) == n_events
        
        print("\n✅ All checks passed! DataLoader successfully reads RootPreprocessor format with LoadConfig.")
        return True


def test_backward_compatibility():
    """Test that DataLoader still works with old structured format."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = os.path.join(tmpdir, "test_data_structured.npz")
        
        # Create structured format (old format)
        n_events = 50
        max_jets = 8
        data = {
            'lepton': np.random.randn(n_events, 2, 4),
            'jet': np.random.randn(n_events, max_jets, 5),
            'assignment_labels': np.random.randint(0, 2, (n_events, max_jets, 2)),
            'event_weight': np.random.uniform(0.5, 1.5, n_events),
        }
        
        np.savez_compressed(npz_path, **data)
        print(f"\nCreated structured NPZ file: {npz_path}")
        
        # Create LoadConfig
        load_config = LoadConfig(
            jet_features=['pt', 'eta', 'phi', 'e', 'b'],
            lepton_features=['pt', 'eta', 'phi', 'e'],
            met_features=[],
            non_training_features=[],
            jet_truth_label='ordered_event_jet_truth_idx',
            lepton_truth_label='event_lepton_truth_idx',
            max_jets=8,
            NUM_LEPTONS=2,
            event_weight='event_weight',
        )
        
        # Load with DataLoader
        print("\nLoading structured data with DataPreprocessor...")
        preprocessor = DataPreprocessor(load_config)
        data_config = preprocessor.load_from_npz(npz_path)
        
        print(f"Data length: {preprocessor.data_length}")
        
        # Verify
        assert preprocessor.feature_data['lepton'].shape == (n_events, 2, 4)
        assert preprocessor.feature_data['jet'].shape == (n_events, max_jets, 5)
        print("✓ Backward compatibility maintained")
        
        print("\n✅ Structured format still works!")
        return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing DataLoader Integration with RootPreprocessor")
    print("=" * 60)
    
    try:
        # Test flat format
        success1 = test_dataloader_flat_format()
        
        # Test backward compatibility
        print("\n" + "=" * 60)
        print("Testing Backward Compatibility")
        print("=" * 60)
        success2 = test_backward_compatibility()
        
        if success1 and success2:
            print("\n" + "=" * 60)
            print("🎉 ALL TESTS PASSED!")
            print("=" * 60)
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
