"""Étape 1 : canaux d'accélération des EDF « mov » de SeizeIT2 -> CSV horodatés (dossier 01_csv_acc)."""
import os
from _commun import arguments, en_parallele, log
from sizeit import config as C, preparation as P

a = arguments(__doc__)
os.makedirs(C.CSV_ACC_DIR, exist_ok=True)
edf = P.runs_mov(C.RAW_DIR, a.patients)
if not edf:
    raise SystemExit(f'aucun EDF trouvé dans {C.RAW_DIR} (variable SIZEIT_RAW_DIR ?)')
res = en_parallele(P.edf_vers_csv, edf, a.jobs, 'EDF -> CSV')
log.info(f'{sum(r is not None for r in res)} CSV écrits dans {C.CSV_ACC_DIR}')
