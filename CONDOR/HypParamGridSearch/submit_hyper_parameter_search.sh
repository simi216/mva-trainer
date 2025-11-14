#!/bin/bash
# HTCondor submission file for hyperparameter grid search
executable = run_training.sh
universe = vanilla

# Resource requirements
RequestGPUs    = 1
Requirements = (GPUs_DeviceName == "Tesla V100-PCIE-32GB")
RequestMemory  = 60000
+RequestRuntime = 21600

# Output files
output = logs/job_$(Cluster)_$(ProcId).out
error = logs/job_$(Cluster)_$(ProcId).err
log = logs/job_$(Cluster)_$(ProcId).log

arguments = $(ARGS)

# Read all columns from params.txt, automatically assigning them to $(ARGS)
queue ARGS from params.txt