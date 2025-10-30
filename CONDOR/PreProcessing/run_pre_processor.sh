#!/bin/bash
# Arguments: $1 = input ROOT file
echo "Running pre-processor on $1 and saving as index $2"
SAV_DIR=/data/dust/group/atlas/ttreco/$3
mkdir -p "$SAV_DIR"
mkdir -p "$SAV_DIR"/temp
echo "./../../preprocessing/bin/merge_root_files.out $SAVE_DIR.root $SAV_DIR/temp/ reco" > merge_command_"$3".sh
~/mva-trainer/preprocessing/bin/run_pre_processes.out "$1" "$SAV_DIR"/temp/temp_"$2".root reco -p nu_flows initial_parton_info