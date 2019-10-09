#!/bin/bash
echo "-_-_-_-_- Rapatriement des poids de Helios -_-_-_-_-"
rsync -zav "$1@helios.calculquebec.ca:PapierFewShot/weights/" "./weights"
echo "-_-_-_-_-_-_-_-_-_-_-   DONE   -_-_-_-_-_-_-_-_-_-_-"
echo ""
echo ""

echo "-_-_-_-_- Rapatriement des poids de Beluga -_-_-_-_-"
rsync -zav "$1@beluga.calculcanada.ca:PapierFewShot/weights/" "./weights"
echo "-_-_-_-_-_-_-_-_-_-_-   DONE   -_-_-_-_-_-_-_-_-_-_-"
echo ""
echo ""
