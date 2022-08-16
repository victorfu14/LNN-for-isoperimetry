#!/bin/bash

#SBATCH --job-name=iso-exp

#SBATCH --mail-user=pbb@umich.edu

#SBATCH --mail-type=END,FAIL

#SBATCH --partition=gpu

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16gb
#SBATCH --gres=gpu:1


#SBATCH --time=1-00:00:00

#SBATCH --account=vvh0

#SBATCH --output=/home/%u/logs/%x-%j.log

source /home/pbb/anaconda3/etc/profile.d/conda.sh
conda activate iso

cd /home/pbb/Project/ISO/

python train_robust.py --conv-layer soc --block-size 4 --dataset cifar10 --n 10000
python train_robust.py --conv-layer soc --block-size 4 --dataset cifar10 --n 8000
python train_robust.py --conv-layer soc --block-size 4 --dataset cifar10 --n 6000
python train_robust.py --conv-layer soc --block-size 4 --dataset cifar10 --n 4000
python train_robust.py --conv-layer soc --block-size 4 --dataset cifar10 --n 2000
python train_robust.py --conv-layer soc --block-size 4 --dataset cifar10 --n 1000
python train_robust.py --conv-layer soc --block-size 4 --dataset cifar10 --n 500
python train_robust.py --conv-layer soc --block-size 4 --dataset cifar10 --n 100

python train_robust.py --conv-layer soc --block-size 4 --dataset cifar100 --n 10000
python train_robust.py --conv-layer soc --block-size 4 --dataset cifar100 --n 8000
python train_robust.py --conv-layer soc --block-size 4 --dataset cifar100 --n 6000
python train_robust.py --conv-layer soc --block-size 4 --dataset cifar100 --n 4000
python train_robust.py --conv-layer soc --block-size 4 --dataset cifar100 --n 2000
python train_robust.py --conv-layer soc --block-size 4 --dataset cifar100 --n 1000
python train_robust.py --conv-layer soc --block-size 4 --dataset cifar100 --n 500
python train_robust.py --conv-layer soc --block-size 4 --dataset cifar100 --n 100

/bin/hostname
sleep 60
