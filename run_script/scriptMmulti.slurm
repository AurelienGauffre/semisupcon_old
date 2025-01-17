#!/bin/bash
#SBATCH --account=cgs@a100

#SBATCH --job-name=Mmulti
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


cd /gpfsscratch/rech/cgs/ued97kp/semisupcon
scancel 736910

python3 train.py --c ./config/configM0.yaml &
python3 train.py --c ./config/configM1.yaml &
python3 train.py --c ./config/configM2.yaml &
python3 train.py --c ./config/configM3.yaml &
python3 train.py --c ./config/configM4.yaml &
python3 train.py --c ./config/configM5.yaml &
python3 train.py --c ./config/configM6.yaml &
python3 train.py --c ./config/configM7.yaml
