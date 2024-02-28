
Understood. You want to create a single SLURM script named auto_scriptE1&E2.slurm, but within this script, you intend to execute two Python scripts concurrently using nohup and sbatch. To achieve this, you'll modify the script to generate a single SLURM file that includes two sbatch commands to run the Python jobs in the background.

Here's how you can modify your script:

bash
Copy code
#!/bin/bash

# Check if at least one argument is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 JOB_NAME [walltime=VALUE]"
    exit 1
fi

# Extract job name from the first argument
JOB_NAME=$1

# Replace '&' with '_' to create the SLURM script filename
SCRIPT_NAME="${JOB_NAME//&/_}"

# Default walltime value
WALLTIME="99"

# If a second argument is provided, extract walltime from it
if [ ! -z "$2" ]; then
    WALLTIME="$2"
fi

# Create the directory for the script if it does not exist
mkdir -p ./run_script

# Create the SLURM script
cat <<EOF >./run_script/auto_script${SCRIPT_NAME}.slurm
#!/bin/bash
#SBATCH --account=cgs@v100
#SBATCH --job-name=$SCRIPT_NAME
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

# Split the JOB_NAME by '&' and run each job in the background
IFS='&' read -ra JOB_PARTS <<< "$JOB_NAME"
for PART in "\${JOB_PARTS[@]}"; do
    nohup python3 train.py --c ./config/config\$PART.yaml &
done
wait

EOF

# Perform git pull to update the repository
git pull

# Submit the SLURM job
sbatch ./run_script/auto_script${SCRIPT_NAME}.slurm