#!/bin/bash
# usage :
# . jean_run.sh E1 50 to run the job E1 with 50 hours of walltime
# . jean_run.sh E1_E2 50 to run the job E1 and E2 on same GPU with 50 hours of walltime

# Check if at least one argument is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 JOB_NAMES [walltime=VALUE]"
    exit 1
fi

# Extract job names from the first argument and split by '_'
IFS='_' read -ra JOB_NAMES <<< "$1"
JOB_NAME_STRING="${JOB_NAMES[*]}"

# Replace spaces with underscores to create the filename
JOB_NAME_FILE="${JOB_NAME_STRING// /_}"

# Default walltime value
WALLTIME="20"

# If a second argument is provided, it is assumed to be the walltime
if [ ! -z "$2" ]; then
    WALLTIME=$2
fi

# Create the directory for the script if it does not exist
mkdir -p ./run_script

# Start creating the SLURM script with a concatenated job name
cat <<EOF > "./run_script/auto_script${JOB_NAME_FILE}.slurm"
#!/bin/bash
#SBATCH --account=cgs@a100
#SBATCH --job-name=${JOB_NAME_FILE}
##SBATCH -C v100-16g                 # uncomment to target only 16GB V100 GPU
##SBATCH -C v100-32g                 # uncomment to target only 32GB V100 GPU
##SBATCH --partition=gpu_p2          # uncomment for gpu_p2 partition (32GB V100 GPU)
#SBATCH -C a100                     # uncomment for gpu_p5 partition (80GB A100 GPU)
#SBATCH --nodes=1                    # we request one node
#SBATCH --ntasks-per-node=1          # with one task per node (= number of GPUs here)
#SBATCH --gres=gpu:1                 # number of GPUs per node (max 8 with gpu_p2, gpu_p4, gpu_p5)

#SBATCH --cpus-per-task=10           # number of cores per task (1/4 of the 4-GPUs node)

#SBATCH --hint=nomultithread         # hyperthreading is deactivated

EOF



cat <<EOF >> "./run_script/auto_script${JOB_NAME_FILE}.slurm"
#SBATCH --time=${WALLTIME}:00:00     # maximum execution time requested (HH:MM:SS)
#SBATCH --output=OAR.%j.${JOB_NAME_FILE}.out    # name of output file

module purge
module load cpuarch/amd
module load pytorch-gpu/py3/1.13.0

cd /gpfsscratch/rech/cgs/ued97kp/semisupcon

EOF

# Append the nohup commands for each job to the SLURM script
for JOB_NAME in "${JOB_NAMES[@]}"; do
    echo "nohup python3 train.py --c ./config/config${JOB_NAME}.yaml &" >> "./run_script/auto_script${JOB_NAME_FILE}.slurm"
done

# Add a wait command to ensure all background jobs complete before the script ends
echo "wait" >> "./run_script/auto_script${JOB_NAME_FILE}.slurm"

# Perform git pull to update the repository
git pull

echo "SLURM script created: ./run_script/auto_script${JOB_NAME_FILE}.slurm"
sbatch "./run_script/auto_script${JOB_NAME_FILE}.slurm"



