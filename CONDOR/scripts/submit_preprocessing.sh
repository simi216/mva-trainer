executable = /afs/desy.de/user/a/aulich/mva_trainer/CONDOR/scripts/run_pre_processor.sh
universe = vanilla
arguments      = $(filename)
output         = logs/$(ClusterId).$(ProcId).out
error          = logs/$(ClusterId).$(ProcId).err
log            = logs/$(ClusterId).$(ProcId).log
RequestMemory = 20000
RequestDisk = 20000
+RequestRuntime = 100000
+MaxRuntime = 100000
transfer_executable = False
should_transfer_files = False

queue filename from input_root_file_list.txt