#!/bin/bash
# HTCondor submission file for hyperparameter grid search

executable = run_training.sh
universe = vanilla

# Resource requirements
RequestCPUs    = 8
RequestGPUs    = 1
RequestMemory  = 20000
+RequestRuntime = 400000
+MaxRuntime     = 400000

# Use GPU if available (optional, adjust based on your needs)
# request_gpus = 1

# Output files - create separate directories for each job
output = logs/job_$(Cluster)_$(hidden_dim)_$(num_layers).out
error = logs/job_$(Cluster)_$(hidden_dim)_$(num_layers).err
log = logs/job_$(Cluster)_$(hidden_dim)_$(num_layers).log

arguments = $(hidden_dim) $(num_layers)

# Grid search parameters
# Queue a job for each combination of hidden_dim and num_layers
queue hidden_dim, num_layers from (
    32, 4
    32, 6
    32, 8
    64, 4
    64, 6
    64, 8
    128, 4
    128, 6
    128, 8
    256, 4
    256, 6
    256, 8
)

# Alternative: If you want to pass other hyperparameters too
# Uncomment and modify as needed:
# queue hidden_dim, num_layers, num_heads, dropout_rate from (
#     64, 4, 8, 0.1
#     64, 6, 8, 0.1
#     128, 4, 8, 0.1
#     128, 6, 8, 0.2
# )