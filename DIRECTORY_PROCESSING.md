# Directory Processing Feature

The preprocessing pipeline now supports processing multiple ROOT files from a directory in a single command.

## Quick Examples

### Merge Multiple Files into One
```bash
# All files combined into single output
python scripts/preprocess_root.py root_files/ merged_data.npz --format npz --merge
```

### Process Files Separately
```bash
# Each file processed independently
python scripts/preprocess_root.py root_files/ processed_files/ --format npz --no-merge
```

### Custom File Pattern
```bash
# Only process files matching pattern
python scripts/preprocess_root.py root_files/ output.npz --pattern "ttbar_*.root" --merge
```

## Features

- **Automatic directory detection**: Just pass a directory path instead of a file
- **Merge option**: Combine all files into a single output (default when output is a file)
- **Separate processing**: Keep files separate (default when output is a directory)
- **Custom patterns**: Use glob patterns to select specific files (e.g., `data_*.root`, `*_mc.root`)
- **Error handling**: Failed files are skipped with warnings, processing continues
- **Progress tracking**: Shows which file is being processed and overall statistics

## Usage Patterns

### 1. Merge for Training
Combine all data into one file for ML training:
```bash
python scripts/preprocess_root.py \
    /data/ttbar_samples/ \
    training_data.npz \
    --format npz \
    --merge \
    --nu-flows
```

### 2. Separate for Analysis
Process each file separately to analyze individual samples:
```bash
python scripts/preprocess_root.py \
    /data/ttbar_samples/ \
    /data/processed_samples/ \
    --format npz \
    --no-merge
```

### 3. Selective Processing
Process only specific files:
```bash
python scripts/preprocess_root.py \
    /data/all_samples/ \
    ttbar_only.npz \
    --pattern "*ttbar*.root" \
    --merge
```

## Python API

```python
from core.RootPreprocessor import preprocess_root_directory

# Merge all files
merged_data = preprocess_root_directory(
    input_dir="/path/to/root_files/",
    output_path="merged_output.npz",
    tree_name="reco",
    output_format="npz",
    merge=True,
    pattern="*.root",
    save_nu_flows=True,
    verbose=True
)

print(f"Processed {len(merged_data['lep_pt'])} total events")

# Process separately
preprocess_root_directory(
    input_dir="/path/to/root_files/",
    output_path="/path/to/output_dir/",
    tree_name="reco",
    output_format="npz",
    merge=False,
    pattern="data_*.root",
    verbose=True
)
```

## Default Behavior

The merge behavior is automatically inferred:

- **Output is a file** (e.g., `output.npz`): `--merge` is default
- **Output is a directory** (e.g., `output_dir/`): `--no-merge` is default

You can always override with explicit `--merge` or `--no-merge` flags.

## Error Handling

If a file fails to process:
- A warning is printed with the error
- Processing continues with remaining files
- Final summary shows successfully processed files

Example output:
```
Found 10 ROOT files to process

Processing file 1/10: sample_001.root
✓ Processed 1234 events

Processing file 2/10: sample_002.root
✗ Warning: Failed to process sample_002.root: Invalid tree structure
...

Successfully processed 9/10 files
Total events: 11106
```

## Performance

- **Parallel I/O**: Each file is read independently (potential for future parallelization)
- **Memory efficient**: When merging, files are processed sequentially
- **Fast output**: NPZ format provides 10-100x faster I/O than ROOT

## See Also

- `scripts/README_PREPROCESSING.md` - Complete preprocessing documentation
- `notebooks/PreprocessingExample.ipynb` - Interactive examples
- `README.md` - Main project documentation
