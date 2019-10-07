#!/bin/bash
echo "Chargement des rnns de Helios"
rsync -zav "piersnel@helios.calculquebec.ca:PapierFewShot/wandb/" "/mnt/Documents/Projets/PapierFewShot/wandb"

echo "Chargement des runs de Beluga"
rsync -zav "piersnel@beluga.calculcanada.ca:PapierFewShot/wandb/" "/mnt/Documents/Projets/PapierFewShot/wandb"
