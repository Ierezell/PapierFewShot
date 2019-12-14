#!/bin/bash
#PBS -N Train_FewShot
#PBS -A syi-200-aa
#PBS -l walltime=10:00:00
#PBS -l nodes=1:gpus=4
#PBS -l feature=k80
###PBS -o ~/PapierFewShot/00trainFewShot.out
###PBS -e ~/PapierFewShot/00trainFewShot.err

module load python/3.7

cd "${PBS_O_WORKDIR}"

# echo "" > ~/PapierFewShot/00trainFewShot.out
# echo "" > ~/PapierFewShot/00trainFewShot.err

source ~/Pytorch_FewShot_Env/bin/activate

python train_ldmk.py
