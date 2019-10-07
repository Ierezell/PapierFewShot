#!/bin/bash
echo "Chargement du code du GAN"
rsync -zav --delete-excluded --include="weights***" --include="wandb***" --include="dataset***" --include="*.py" --include="*.yaml" --exclude="*" "/mnt/Documents/Projets/PapierFewShot/" "piersnel@beluga.calculcanada.ca:PapierFewShot/"
echo "Chargement du code du RNN"
rsync -zav --delete-excluded --include="weights***" --include="wandb***" --include="dataset***" --include="*.py" --include="*.yaml" --exclude="*" "/mnt/Documents/Projets/SacadeDetect/" "piersnel@beluga.calculcanada.ca:SacadeDetect/"
