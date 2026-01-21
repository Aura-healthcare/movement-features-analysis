
#!/usr/bin/env bash

set -euo pipefail

# Log file horodaté (format: 2026-01-15_14-23-05)

ts="$(date +'%Y-%m-%d_%H-%M-%S')"
log="logs/features_${ts}.log" 
:> "$log"   # nettoye le fichier s'il existe et envoie dans ce fichier

IN_DIR="/home/aura-fabien/movement-features-analysis/datas/datas_csv_acc"
OUT_DIR="/home/aura-fabien/movement-features-analysis/features/computation_15_01"


mkdir -p "$OUT_DIR" # fait la dir s'il nexiste pas

for in_file in "$IN_DIR"/*.csv; do # / c'est parec que on est en bash
  base="$(basename "$in_file" .csv)" # basename c'est le nom sans l'extension ni le path
  out_file="$OUT_DIR/features_${base}.csv" #
  echo "[START] $in_file -> $out_file" >>"$log" 2>&1
  python3  features_computation.py -i "$in_file" -o "$out_file" >>"$log" 2>&1 &

done # bash fermer les boucles

wait
echo "Tous les processus toto.py sont terminés"