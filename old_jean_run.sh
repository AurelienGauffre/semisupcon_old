#!/bin/bash

# Check if at least one argument is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 JOB_NAME [walltime=VALUE]"
    exit 1
fi

# Extract job name from the first argument and split by underscore
JOB_NAME=$1
IFS='_' read -ra JOB_PARTS <<< "$JOB_NAME"

# Default walltime value
WALLTIME="99"

# If a second argument is provided, extract walltime from it
if [ ! -z "$2" ]; then
    WALLTIME=$2
fi

# Create the directory for the script if it does not exist
mkdir -p ./run_script

# Create the SLURM script
cat <<EOF >./run_script/auto_script${JOB_NAME}.slurm
#!/bin/bash
#SBATCH --account=cgs@v100
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=gpu_p2

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu-t4
#SBATCH --time=${WALLTIME}:00:00
#SBATCH --output=OAR.%j.out

module purge
module load pytorch-gpu/py3/1.13.0

cd /gpfsscratch/rech/cgs/ued97kp/semisupcon
EOF

# Append sbatch commands for each job part
for PART in "${JOB_PARTS[@]}"; do
    echo "sbatch ./run_script/auto_script${PART}.slurm" >> ./run_script/auto_script${JOB_NAME}.slurm
done

# Perform git pull to update the repository
git pull

# Submit the SLURM job
sbatch ./run_script/auto_script${JOB_NAME}.slurm
