#!/usr/bin/env bash
# Enchaîne tout le pipeline. Usage : bash lancer_tout.sh [--liste ../patients_developpement.txt] [--jobs 8]
# Chaque étape reprend là où elle s'est arrêtée (les fichiers déjà produits ne sont pas recalculés).
set -euo pipefail
cd "$(dirname "$0")"
for etape in 01_edf_vers_csv 02_renormalisation 03_features_acc 04_fusion_grille 05_preparation_patients \
             06_decoupage 07_evaluation 08_entrainer_modeles; do
    echo "==================== $etape"
    python "$etape.py" "$@"
done
