#!/bin/bash

#SBATCH --job-name=iso-exp

#SBATCH --mail-user=pbb@umich.edu

#SBATCH --mail-type=END,FAIL

#SBATCH --partition=standard

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16gb

#SBATCH --time=3-00:00:00

#SBATCH --account=vvh

#SBATCH --output=/home/%u/logs/%x-%j.log

source /home/pbb/anaconda3/etc/profile.d/conda.sh
conda activate iso

cd /home/pbb/Project/ISO/

python train_robust.py --conv-layer CONV --activation ACT --num-blocks BLOCKS --dataset DATASET --gamma GAMMA

/bin/hostname
sleep 60
