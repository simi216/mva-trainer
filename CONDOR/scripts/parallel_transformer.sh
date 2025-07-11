executable = /afs/desy.de/user/a/aulich/mva_trainer/CONDOR/submitCondor.sh
universe = vanilla
arguments = "python3 transformer_script.py $(arg1)"
error = /afs/desy.de/user/a/aulich/mva_trainer/CONDOR/logs/$(ClusterId).$(ProcId).err
log = /afs/desy.de/user/a/aulich/mva_trainer/CONDOR/logs/$(ClusterId).$(ProcId).log
output = /afs/desy.de/user/a/aulich/mva_trainer/CONDOR/logs/$(ClusterId).$(ProcId).out
RequestCPUs = 20
RequestMemory = 40000
+RequestRuntime = 100000
+MaxRuntime = 100000
transfer_executable = False
should_transfer_files = False

queue arg1 from -
