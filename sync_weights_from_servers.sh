#!/bin/bash
echo "Chargement des poids de Helios"
rsync -zav "$1@helios.calculquebec.ca:PapierFewShot/weights/" "./weights"

echo "Chargement des poids de Beluga"
rsync -zav "$1@beluga.calculcanada.ca:PapierFewShot/weights/" "./weights"
