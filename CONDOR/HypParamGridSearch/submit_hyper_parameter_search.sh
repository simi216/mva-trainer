#!/bin/bash
# HTCondor submission file for hyperparameter grid search
executable = run_training.sh
universe = vanilla

# Resource requirements
RequestGPUs    = 1
Requirements = (GPUs_DeviceName == "Tesla V100-PCIE-32GB")
RequestMemory  = 60000
RequestRuntime = 7200

# Output files
output = logs/job_$(Cluster)_$(architecture)_$(hidden_dim)_$(num_layers)_$(num_heads).out
error = logs/job_$(Cluster)_$(architecture)_$(hidden_dim)_$(num_layers)_$(num_heads).err
log = logs/job_$(Cluster)_$(architecture)_$(hidden_dim)_$(num_layers)_$(num_heads).log

arguments = $(hidden_dim) $(num_layers) $(num_heads) $(architecture)

# Queue from file - reads each line and assigns to variables
queue hidden_dim, num_layers, num_heads, architecture from params.txt