"""Étape 5 : un fichier par patient (05_patients/<patient>.parquet), avec l'identifiant de chaque crise (seizure_id)."""
import glob
import os
from _commun import arguments, en_parallele, log
from sizeit import config as C, preparation as P

a = arguments(__doc__)
os.makedirs(C.DATA_DIR, exist_ok=True)
pids = sorted({os.path.basename(f).split('_')[0] for f in glob.glob(os.path.join(C.GRILLE_DIR, 'sub-*.parquet'))})
if a.patients:
    pids = [p for p in pids if p in a.patients]
res = en_parallele(P.preparer_patient, pids, a.jobs, 'préparation des patients')
log.info(f'{sum(r is not None for r in res)} patients écrits dans {C.DATA_DIR}')
