#!/bin/bash
# HTCondor submission file for hyperparameter grid search

executable = run_training.sh
universe = vanilla
# Resource requirements
RequestCPUs    = 20
RequestMemory  = 20000
+RequestRuntime = 400000
+MaxRuntime     = 400000

# Use GPU if available (optional, adjust based on your needs)
# request_gpus = 1

# Output files - create separate directories for each job
output = logs/job_$(Cluster)_$(architecture)_$(hidden_dim)_$(num_layers).out
error = logs/job_$(Cluster)_$(architecture)_$(hidden_dim)_$(num_layers).err
log = logs/job_$(Cluster)_$(architecture)_$(hidden_dim)_$(num_layers).log

# Define hyperparameter grid
architectures = FeatureConcatTransformer FeatureConcatRNN CrossAttentionTransformer
hidden_dims   = 32 64 128
num_layers    = 4 6 8



# Cartesian product (each combination)
queue architecture in $(architectures) \
      hidden_dim in $(hidden_dims) \
      num_layers in $(num_layers)