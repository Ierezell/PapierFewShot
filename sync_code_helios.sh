#!/bin/bash
echo "Chargement du code du GAN"
rsync -zav --delete-excluded --include="weights***" --include="wandb***" --include="dataset***" --include="*.py" --include="*.yaml" --exclude="*" "/mnt/Documents/Projets/PapierFewShot/" "piersnel@helios.calculquebec.ca:PapierFewShot/"
echo "Chargement du code du RNN"
rsync -zav --delete-excluded --include="weights***" --include="wandb***" --include="dataset***" --include="*.py" --include="*.yaml" --exclude="*" "/mnt/Documents/Projets/SacadeDetect/" "piersnel@helios.calculquebec.ca:SacadeDetect/"
