#!/bin/bash

# Check if at least one argument is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 JOB_NAME [walltime=VALUE]"
    exit 1
fi

# Extract job name from the first argument
JOB_NAME=$1

# Default walltime value
WALLTIME="99"

# If a second argument is provided, extract walltime from it
if [ ! -z "$2" ]; then
    WALLTIME=$2
fi

# Create the directory for the script if it does not exist
mkdir -p ./run_script_OAR
touch ./run_script_OAR/auto_runoar${JOB_NAME}.sh
chmod +x ./run_script_OAR/auto_runoar${JOB_NAME}.sh
git pull
# Create the SLURM script
cat <<EOF >./run_script_OAR/auto_runoar${JOB_NAME}.sh
cd ~/semisupcon
. envsemisupcon/bin/activate
git pull
DSDIR="/home/aptikal/gauffrea/datasets"
DSDIR_CUSTOM="/home/aptikal/gauffrea/datasets"
export DSDIR
export DSDIR_CUSTOM
python3 train.py --c ./config/config${JOB_NAME}.yaml
EOF



oarsub -l /host=1/gpu=1,walltime=${WALLTIME}:0:0 /home/aptikal/gauffrea/semisupcon/run_script_OAR/auto_runoar${JOB_NAME}.sh
