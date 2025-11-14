# Python Preprocessing

This directory contains the Python-based preprocessing pipeline that replaces the previous C++ implementation.

## Overview

The preprocessing pipeline performs the following operations on ROOT files containing particle physics event data:

1. **Event Pre-selection**: Filters events based on physics requirements (lepton/jet multiplicities, charge conservation, truth matching)
2. **Particle Ordering**: Orders leptons by charge and jets by pT
3. **Derived Feature Computation**: Calculates invariant masses, angular separations (ΔR), and other kinematic variables
4. **Truth Information Extraction**: Extracts truth-level top quark, neutrino, and ttbar system information
5. **Optional Features**: Can include NuFlow neutrino reconstruction results and initial parton information

## Usage

### Command Line

The simplest way to preprocess ROOT files is using the command-line script:

#### Single File Processing

```bash
# Preprocess to ROOT format
python scripts/preprocess_root.py input.root output.root --tree reco

# Preprocess to NPZ format (recommended for ML workflows)
python scripts/preprocess_root.py input.root output.npz --format npz

# Include NuFlow results
python scripts/preprocess_root.py input.root output.npz --format npz --nu-flows

# Include initial parton information
python scripts/preprocess_root.py input.root output.root --initial-parton-info

# Combine options
python scripts/preprocess_root.py input.root output.npz --format npz --nu-flows --initial-parton-info
```

#### Directory Processing

Process multiple ROOT files at once:

```bash
# Merge all files in directory into one output
python scripts/preprocess_root.py input_dir/ merged_output.npz --format npz --merge

# Process files separately (creates output_dir/file1_processed.npz, etc.)
python scripts/preprocess_root.py input_dir/ output_dir/ --format npz --no-merge

# Use custom file pattern
python scripts/preprocess_root.py input_dir/ output.npz --pattern "data_*.root" --merge

# Process directory with all options
python scripts/preprocess_root.py input_dir/ output.npz --format npz --merge --nu-flows --initial-parton-info
```

**Note**: When processing directories:
- `--merge` (default if output is a file): Combines all ROOT files into a single output file
- `--no-merge` (default if output is a directory): Creates separate output files for each input file
- `--pattern`: Glob pattern for selecting files (default: `*.root`)


### Python API

You can also use the preprocessing functionality directly in Python:

#### Single File

```python
from core.RootPreprocessor import preprocess_root_file

# Simple preprocessing
data = preprocess_root_file(
    input_path="input.root",
    output_path="output.npz",
    tree_name="reco",
    output_format="npz"
)

# With options
data = preprocess_root_file(
    input_path="input.root",
    output_path="output.npz",
    tree_name="reco",
    output_format="npz",
    save_nu_flows=True,
    save_initial_parton_info=True,
    verbose=True
)
```

#### Multiple Files (Directory)

```python
from core.RootPreprocessor import preprocess_root_directory

# Merge all files into one output
merged_data = preprocess_root_directory(
    input_dir="path/to/root_files/",
    output_path="merged_output.npz",
    tree_name="reco",
    output_format="npz",
    merge=True,  # Combine all files
    pattern="*.root",
    save_nu_flows=True,
    verbose=True
)

# Process files separately
preprocess_root_directory(
    input_dir="path/to/root_files/",
    output_path="output_directory/",
    tree_name="reco",
    output_format="npz",
    merge=False,  # Keep files separate
    pattern="data_*.root",  # Custom pattern
    verbose=True
)
```


### Advanced Usage

For more control, use the `RootPreprocessor` class directly:

```python
from core.RootPreprocessor import RootPreprocessor, PreprocessorConfig

config = PreprocessorConfig(
    input_path="input.root",
    output_path="output.npz",
    tree_name="reco",
    output_format="npz",
    save_nu_flows=True,
    save_initial_parton_info=True,
    verbose=True
)

preprocessor = RootPreprocessor(config)
preprocessor.process()

# Access processed data
data = preprocessor.get_processed_data()
```

## Integration with DataPreprocessor

The preprocessed data (especially in NPZ format) can be loaded directly into the existing `DataPreprocessor` workflow:

```python
from core import DataPreprocessor, LoadConfig

# Option 1: Load NPZ file directly
preprocessor = DataPreprocessor(load_config)
preprocessor.load_from_npz("preprocessed_data.npz")

# Option 2: Traditional ROOT loading (works with preprocessed ROOT files)
preprocessor.load_data("preprocessed_data.root", "reco")

# Save processed data to NPZ for fast loading later
preprocessor.save_to_npz("processed_for_ml.npz")
```

## Output Format Comparison

### ROOT Format
- **Pros**: Compatible with existing ROOT-based workflows, can be inspected with ROOT tools
- **Cons**: Slower I/O, larger file sizes
- **Use when**: Need compatibility with ROOT ecosystem

### NPZ Format  
- **Pros**: Fast I/O (10-100x faster), smaller file sizes with compression, easier to work with in Python
- **Cons**: Not compatible with ROOT tools
- **Use when**: Pure Python/ML workflow (recommended)

## Features Produced

### Standard Features
- `lep_pt`, `lep_eta`, `lep_phi`, `lep_e`: Lepton kinematics (ordered by charge)
- `lep_charge`, `lep_pid`: Lepton charge and PDG ID
- `event_lepton_truth_idx`: Truth matching indices
- `ordered_jet_pt`, `ordered_jet_eta`, `ordered_jet_phi`, `ordered_jet_e`: Jet kinematics (ordered by pT)
- `ordered_jet_b_tag`: B-tagging scores
- `ordered_event_jet_truth_idx`: Truth matching indices
- `N_jets`: Number of jets per event

### Derived Features
- `m_l1j`, `m_l2j`: Invariant masses of lepton-jet systems
- `dR_l1j`, `dR_l2j`: Angular separation (ΔR) between leptons and jets
- `dR_l1l2`: Angular separation between leptons

### Truth Information
- `truth_ttbar_mass`, `truth_ttbar_pt`: ttbar system kinematics
- `truth_tt_boost_parameter`: Boost parameter of ttbar system
- `truth_top_*`: Top quark kinematics (mass, pt, eta, phi, e)
- `truth_tbar_*`: Anti-top quark kinematics
- `truth_*_neutino_*`: Neutrino kinematics (both 4-vector and momentum components)

### Optional: NuFlow Results (with `--nu-flows`)
- `nu_flows_neutrino_px`, `nu_flows_neutrino_py`, `nu_flows_neutrino_pz`
- `nu_flows_antineutrino_px`, `nu_flows_antineutrino_py`, `nu_flows_antineutrino_pz`

### Optional: Initial Parton Info (with `--initial-parton-info`)
- `truth_initial_parton_num_gluons`: Number of gluons in initial state

## Pre-selection Criteria

Events must pass the following cuts:

1. Exactly 2 leptons (electrons or muons)
2. At least 2 jets
3. Valid truth matching information (≥6 truth indices)
4. Valid b-jet truth indices (0 or 3) within allowed range
5. Valid lepton truth indices
6. Opposite-sign lepton charges
7. Total charge = 0

## Performance

Typical processing speed: **~1000-5000 events/second** (depends on number of jets per event)

Example timing for 100k events:
- C++ preprocessing: ~30-60 seconds
- Python preprocessing: ~20-40 seconds (comparable or faster!)
- NPZ I/O: ~1-2 seconds
- ROOT I/O: ~10-30 seconds

## Migration from C++

The Python implementation provides identical functionality to the previous C++ preprocessor:

| C++ Flag | Python Equivalent |
|----------|------------------|
| `-p nu_flows` | `--nu-flows` |
| `-p initial_parton_info` | `--initial-parton-info` |

Example migration:

```bash
# Old C++ command
./run_pre_processes input.root output.root reco -p nu_flows initial_parton_info

# New Python command
python scripts/preprocess_root.py input.root output.root --tree reco --nu-flows --initial-parton-info
```

## Advantages of Python Implementation

1. **No compilation required**: Works immediately without Make/C++ toolchain
2. **Easier to modify**: Pure Python code is more accessible for physics users
3. **Better integration**: Seamlessly integrates with Python ML pipeline
4. **Modern I/O**: Supports efficient NPZ format
5. **Better error handling**: More informative error messages
6. **Cross-platform**: Works on any system with Python (no ROOT compilation issues)

## Requirements

- Python ≥ 3.8
- numpy
- uproot
- awkward
