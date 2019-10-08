#!/bin/bash
echo "Chargement des rnns de Helios"
rsync -zav "$1@helios.calculquebec.ca:PapierFewShot/wandb/" "./wandb"

echo "Chargement des runs de Beluga"
rsync -zav "$1@beluga.calculcanada.ca:PapierFewShot/wandb/" "./wandb"
