#!/bin/bash

# HTCondor wrapper script for training
# This script is called by HTCondor with environment variables

hidden_dim=$1
num_layers=$2
num_heads=$3
architecture=$4

# Print environment info for debugging
echo "Starting training job..."
echo "Hidden dim: $hidden_dim"
echo "Num layers: $num_layers"
echo "Working directory: $(pwd)"
echo "Date: $(date)"

# Activate your Python environment (adjust path as needed)
source /data/dust/group/atlas/ttreco/venv/bin/activate
# OR if using conda:
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate your_env_name

# Change to the script directory
cd /afs/desy.de/user/a/aulich/mva-trainer/CONDOR/HypParamGridSearch || exit 1

# Create logs directory if it doesn't exist
#mkdir -p logs

# Run the Python script with the hyperparameters
python train_hyperparameter.py \
    --hidden_dim "$hidden_dim" \
    --num_layers "$num_layers" \
    --num_heads  "$num_heads"\
    --architecture "$architecture" \
    --data_config "/afs/desy.de/user/a/aulich/mva-trainer/CONDOR/workspace_config.yaml" \
    --dropout_rate 0.1 \
    --learning_rate 1e-4 \
    --weight_decay 1e-4 \
    --batch_size 2048 \
    --epochs 50 \
    --patience 10 \
    --max_events 10000000 \
    --root_dir "/afs/desy.de/user/a/aulich/mva-trainer/" \
    --data_type "nominal"


# Check if the script succeeded
exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code: $exit_code"
fi

#exit $exit_code