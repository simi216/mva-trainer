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

arguments = $(hidden_dim) $(num_layers) $(architecture)

# Grid search parameters
# Queue a job for each combination of hidden_dim and num_layers
queue architecture, hidden_dim, num_layers from (
    FeatureConcatTransformer, 32, 4
    FeatureConcatTransformer, 32, 6
    FeatureConcatTransformer, 32, 8
    FeatureConcatTransformer, 64, 4
    FeatureConcatTransformer, 64, 6
    FeatureConcatTransformer, 64, 8
    FeatureConcatTransformer, 128, 4
    FeatureConcatTransformer, 128, 6
    FeatureConcatTransformer, 128, 8
    FeatureConcatRNN, 32, 4
    FeatureConcatRNN, 32, 6
    FeatureConcatRNN, 32, 8
    FeatureConcatRNN, 64, 4
    FeatureConcatRNN, 64, 6
    FeatureConcatRNN, 64, 8
    FeatureConcatRNN, 128, 4
    FeatureConcatRNN, 128, 6
    FeatureConcatRNN, 128, 8
    CrossAttentionTransformer, 32, 4
    CrossAttentionTransformer, 32, 6
    CrossAttentionTransformer, 32, 8
    CrossAttentionTransformer, 64, 4
    CrossAttentionTransformer, 64, 6
    CrossAttentionTransformer, 64, 8
    CrossAttentionTransformer, 128, 4
    CrossAttentionTransformer, 128, 6
    CrossAttentionTransformer, 128, 8
)