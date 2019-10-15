#!/bin/bash
echo "-_-_-_- Chargement du code du GAN sur Beluga -_-_-_-"
rsync -zav --delete-excluded --include="weights***" --include="wandb***" --include="dataset***" --include="*.py" --include="*.yaml" --include="*.sh" --exclude="*" "./" "$1@beluga.calculcanada.ca:PapierFewShot/"
echo "-_-_-_-_-_-_-_-_-_-_-   DONE   -_-_-_-_-_-_-_-_-_-_-"
echo ""
echo ""
