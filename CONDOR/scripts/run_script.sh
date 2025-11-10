#!/bin/bash
# HTCondor submission file for hyperparameter grid search
executable = submitCondor.sh
universe = vanilla

# Resource requirements
RequestCPUs    = 20
RequestMemory  = 40000
+RequestRuntime = 800000
+MaxRuntime     = 800000

# Output files
output = logs/job_$(Cluster).out
error = logs/job_$(Cluster).err
log = logs/job_$(Cluster).log

arguments = python3 scripts/train_regression_transformer.py \
            --data_config workspace_config.yaml

# Queue from file - reads each line and assigns to variables
queue 1