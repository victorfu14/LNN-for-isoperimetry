#!/bin/bash

#SBATCH --job-name=cifar10

#SBATCH --mail-user=pbb@umich.edu

#SBATCH --mail-type=END,FAIL

#SBATCH --partition=gpu

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16gb
#SBATCH --gres=gpu:1


#SBATCH --time=3-00:00:00

#SBATCH --account=vvh0

#SBATCH --output=/home/%u/Project/ISO/logs/%x-%j.log

source /home/pbb/anaconda3/etc/profile.d/conda.sh
conda activate iso

cd /home/pbb/Project/ISO/

python train.py --conv-layer soc --block-size 2 --dataset cifar10 --lr-max 0.2
python eval.py --conv-layer soc --block-size 2 --dataset cifar10

/bin/hostname
sleep 60
