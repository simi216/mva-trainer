#!/bin/bash
# HTCondor submission file for hyperparameter grid search

executable = run_training.sh
universe = vanilla

# Resource requirements
RequestCPUs    = 20
RequestMemory  = 60000
+RequestRuntime = 1000000
+MaxRuntime     = 1000000

# Output files (generic naming)
output = logs/job_$(Cluster)_$(Process).out
error  = logs/job_$(Cluster)_$(Process).err
log    = logs/job_$(Cluster)_$(Process).log

# Automatically pass all columns from params.txt to your script
# Each line in params.txt becomes a set of arguments passed to run_training.sh
arguments = $(ARGS)

# Read all columns from params.txt, automatically assigning them to $(ARGS)
queue ARGS from params.txt
