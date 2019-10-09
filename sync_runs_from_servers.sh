#!/bin/bash
echo "-_-_-_- Rapatriement des runs GAN de Helios -_-_-_-"
rsync -zav "$1@helios.calculquebec.ca:PapierFewShot/wandb/" "./wandb"
echo "-_-_-_-_-_-_-_-_-_-_-   DONE   -_-_-_-_-_-_-_-_-_-_-"
echo ""
echo ""

echo "-_-_-_- Rapatriement des runs GAN de Beluga -_-_-_-"
rsync -zav "$1@beluga.calculcanada.ca:PapierFewShot/wandb/" "./wandb"
echo "-_-_-_-_-_-_-_-_-_-_-   DONE   -_-_-_-_-_-_-_-_-_-_-"
echo ""
echo ""
