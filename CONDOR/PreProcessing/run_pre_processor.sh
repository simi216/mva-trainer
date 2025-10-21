#!/bin/bash
# Arguments: $1 = input ROOT file
echo "Running pre-processor on $1 and saving as index $2"
SAV_DIR=/data/dust/group/atlas/ttreco/toponium_full_training
mkdir -p "$SAV_DIR"
mkdir -p "$SAV_DIR"/temp
~/mva-trainer/preprocessing/bin/run_pre_processes.out "$1" "$SAV_DIR"/temp/temp_"$2".root reco -p nu_flows initial_parton_info