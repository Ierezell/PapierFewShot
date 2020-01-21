#!/bin/bash
#SBATCH --time=71:59:00
#SBATCH --account=def-mparizea
#SBATCH --job-name=TrainFewShot
#SBATCH --gres=gpu:4
#SBATCH --mem=32000M
###SBATCH --output=/home/piersnel/PapierFewShot/00trainFewShot.out

###echo "" > /home/piersnel/PapierFewShot/00trainFewShot.out

module load python/3.7

source ~/Pytorch_FewShot_Env/bin/activate

python train_ldmk.py
