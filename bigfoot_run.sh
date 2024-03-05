#!/bin/bash

# Check if at least one argument is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 JOB_NAME [walltime=VALUE]"
    exit 1
fi

# Extract job names from the first argument and split by '_'
IFS='_' read -ra JOB_NAMES <<< "$1"
JOB_NAME_STRING="${JOB_NAMES[*]}"

# Replace spaces with underscores to create the filename
JOB_NAME_FILE="${JOB_NAME_STRING// /_}"

# Default walltime value
WALLTIME="48"

# If a second argument is provided, extract walltime from it
if [ ! -z "$2" ]; then
    WALLTIME=$2
fi

# Create the directory for the script if it does not exist
mkdir -p ./run_script_OAR
touch ./run_script_OAR/auto_runoar_bigfoot_${JOB_NAME_FILE}.sh
chmod +x ./run_script_OAR/auto_runoar_bigfoot_${JOB_NAME_FILE}.sh
git pull
# Create the SLURM script
cat <<EOF >./run_script_OAR/auto_runoar_bigfoot_${JOB_NAME_FILE}.sh
source /applis/environments/conda.sh
conda activate pytorch_stable
cd ~/semisupcon
git pull
EOF

# Append the nohup commands for each job to the SLURM script
for JOB_NAME in "${JOB_NAMES[@]}"; do
    echo "python3 train.py --c ./config/config${JOB_NAME}.yaml" >> "./run_script_OAR/auto_runoar_bigfoot_${JOB_NAME_FILE}.sh"
done

#oarsub -l /host=1/gpu=1,walltime=${WALLTIME}:0:0 /home/aptikal/gauffrea/semisupcon/run_script_OAR/auto_runoar_bigfoot_${JOB_NAME_FILE}.sh
oarsub -l /nodes=1/gpu=1,walltime=${WALLTIME}:0:0 --project pr-lawbot -p "gpumodel='V100'" "nvidia-smi -L" -S /home/gauffrea/semisupcon/run_script_OAR/auto_runoar_bigfoot_${JOB_NAME_FILE}.sh
