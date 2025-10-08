executable = /afs/desy.de/user/a/aulich/mva-trainer/CONDOR/submitCondor.sh
universe   = vanilla

# run python with the script path as argument (submit-variable 'script' defined by queue)
arguments  = "python3 $(script)"

# use $BASENAME(script) to strip directories from the queued value
error  = /afs/desy.de/user/a/aulich/mva-trainer/CONDOR/logs/$(Cluster).$BASENAME(script).err
output = /afs/desy.de/user/a/aulich/mva-trainer/CONDOR/logs/$(Cluster).$BASENAME(script).out
log    = /afs/desy.de/user/a/aulich/mva-trainer/CONDOR/logs/$(Cluster).$BASENAME(script).log

RequestCPUs    = 8
RequestGPUs    = 1
RequestMemory  = 20000
+RequestRuntime = 100000
+MaxRuntime     = 100000

transfer_executable = False
should_transfer_files = False

queue 1