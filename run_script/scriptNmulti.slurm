#!/bin/bash
#SBATCH --account=cgs@a100

#SBATCH --job-name=Nmulti
##SBATCH -C v100-16g                 # uncomment to target only 16GB V100 GPU
##SBATCH -C v100-32g                 # uncomment to target only 32GB V100 GPU
##SBATCH --partition=gpu_p2          # uncomment for gpu_p2 partition (32GB V100 GPU)
##SBATCH --partition=gpu_p4          # uncomment for gpu_p4 partition (40GB A100 GPU) <20h
#SBATCH -C a100                     # uncomment for gpu_p5 partition (80GB A100 GPU)

#SBATCH --nodes=1                    # we request one node
#SBATCH --ntasks-per-node=1          # with one task per node (= number of GPUs here)
#SBATCH --gres=gpu:1                 # number of GPUs per node (max 8 with gpu_p2, gpu_p4, gpu_p5)

#SBATCH --cpus-per-task=10           # number of cores per task (1/4 of the 4-GPUs node)

#SBATCH --hint=nomultithread         # hyperthreading is deactivated
##SBATCH --qos=qos_gpu-t4 #for jobs > 100h
#SBATCH --time=20:00:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --output=OAR.%j.out    # name of output file

module purge  
module load cpuarch/amd
module load pytorch-gpu/py3/1.13.0

scancel 736912
cd /gpfsscratch/rech/cgs/ued97kp/semisupcon

python3 train.py --c ./config/configN1.yaml &
python3 train.py --c ./config/configN2.yaml &
python3 train.py --c ./config/configN3.yaml &
python3 train.py --c ./config/configN4.yaml &
python3 train.py --c ./config/configN5.yaml &
python3 train.py --c ./config/configN6.yaml &
python3 train.py --c ./config/configN7.yaml &
python3 train.py --c ./config/configN8.yaml &
python3 train.py --c ./config/configN9.yaml &
python3 train.py --c ./config/configN10.yaml
