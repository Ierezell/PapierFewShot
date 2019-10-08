#!/bin/bash
echo "Chargement du code du GAN"
rsync -zav --delete-excluded --include="weights***" --include="wandb***" --include="dataset***" --include="*.py" --include="*.yaml" --include="*.sh" --exclude="*" "./" "$1@helios.calculquebec.ca:PapierFewShot/"
