#!/bin/bash

# Usage check
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 JOB_NAMES [walltime=VALUE]"
    exit 1
fi

# Extract job names from the first argument and split by '_'
IFS='_' read -ra JOB_NAMES <<< "$1"

# Default walltime value
WALLTIME="99"

# Extract walltime from the second argument if provided
if [ ! -z "$2" ]; then
    WALLTIME=$2
fi

# Create the directory for the script if it does not exist
mkdir -p ./run_script

# Create the SLURM script with dynamic job names
cat <<EOF >./run_script/auto_script${JOB_NAMES[*]}.slurm
#!/bin/bash
#SBATCH --account=cgs@v100
#SBATCH --job-name=${JOB_NAMES[*]}
##SBATCH -C v100-16g                 # uncomment to target only 16GB V100 GPU
##SBATCH -C v100-32g                 # uncomment to target only 32GB V100 GPU
#SBATCH --partition=gpu_p2          # uncomment for gpu_p2 partition (32GB V100 GPU)

#SBATCH --nodes=1                    # we request one node
#SBATCH --ntasks-per-node=1          # with one task per node (= number of GPUs here)
#SBATCH --gres=gpu:1                 # number of GPUs per node (max 8 with gpu_p2, gpu_p4, gpu_p5)

#SBATCH --cpus-per-task=10           # number of cores per task (1/4 of the 4-GPUs node)

#SBATCH --hint=nomultithread         # hyperthreading is deactivated
#SBATCH --qos=qos_gpu-t4 #for jobs > 100h
#SBATCH --time=${WALLTIME}:00:00     # maximum execution time requested (HH:MM:SS)
#SBATCH --output=OAR.%j.out    # name of output file

module purge
module load pytorch-gpu/py3/1.13.0

cd /gpfsscratch/rech/cgs/ued97kp/semisupcon

EOF

# Append the nohup commands for each job to the SLURM script
for JOB_NAME in "${JOB_NAMES[@]}"; do
    echo "nohup python3 train.py --c ./config/config${JOB_NAME}.yaml &" >> ./run_script/auto_script${JOB_NAMES[*]}.slurm
done

# Add a wait command to ensure all background jobs complete before the script ends
echo "wait" >> ./run_script/auto_script${JOB_NAMES[*]}.slurm

# Perform git pull to update the repository
git pull

# Submit the SLURM job
sbatch ./run_script/auto_script${JOB_NAMES[*]}.slurm
