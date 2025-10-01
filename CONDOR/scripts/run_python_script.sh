executable = /afs/desy.de/user/a/aulich/mva_trainer/CONDOR/submitCondor.sh
universe = vanilla

# $(script) will be substituted when you queue
arguments = "python3 scripts/train_rnn_script.py"

error  = /afs/desy.de/user/a/aulich/mva_trainer/CONDOR/logs/$(script).$(ProcId).err
log    = /afs/desy.de/user/a/aulich/mva_trainer/CONDOR/logs/$(script).$(ProcId).log
output = /afs/desy.de/user/a/aulich/mva_trainer/CONDOR/logs/$(script).$(ProcId).out

RequestCPUs    = 8
#RequestGPUs    = 1
RequestMemory  = 20000
+RequestRuntime = 100000
+MaxRuntime     = 100000

transfer_executable = False
should_transfer_files = False
checkpoint = True

queue 1