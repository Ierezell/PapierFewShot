#!/bin/bash
echo "-_-_-_- Chargement du code du GAN sur Helios -_-_-_-"
rsync -zav --delete-excluded --include="weights***" --include="wandb***" --include="dataset***" --include="*.py" --include="*.yaml" --include="*.sh" --exclude="*" "./" "$1@helios.calculquebec.ca:PapierFewShot/"
echo "-_-_-_-_-_-_-_-_-_-_-   DONE   -_-_-_-_-_-_-_-_-_-_-"
echo ""
echo ""
