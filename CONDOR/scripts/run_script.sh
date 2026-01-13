#!/bin/bash
# HTCondor submission file for hyperparameter grid search
executable = submitCondor.sh
universe = vanilla

# Resource requirements
RequestGPUs    = 1
RequestMemory  = 60000
+RequestRuntime = 43200

# Output files
output = logs/job_$(Cluster).out
error = logs/job_$(Cluster).err
log = logs/job_$(Cluster).log

arguments = python3 scripts/train_regression_transformer.py

# Queue from file - reads each line and assigns to variables
queue 1