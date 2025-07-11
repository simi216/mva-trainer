#!/bin/bash
# Arguments: $1 = input ROOT file
echo "Running pre-processor on $1"
SAV_DIR=/data/dust/group/atlas/ttreco/full_training_new
mkdir -p "$SAV_DIR"
~/mva_trainer/bin/run_pre_processes.out /data/dust/group/atlas/ttreco/mc202507/user.kbehr.410472.PhPy8EG.DAOD_PHYS.e6348_s3681_r13145_r13146_p6697.250627-v1_output/"$1" /data/dust/group/atlas/ttreco/full_training_new/"$1" reco -p nu_flows initial_parton_info