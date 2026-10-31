"""Étape 4 : fusion HRV (fournie) + features ACC + annotations de crises -> une grille à 1 s par run (04_grille)."""
import glob
import os
from _commun import arguments, en_parallele, log
from sizeit import config as C, preparation as P

a = arguments(__doc__)
os.makedirs(C.GRILLE_DIR, exist_ok=True)
runs = sorted(os.path.basename(f)[len('features_ACC_EEG_SD_'):-len('_mov.csv')]
              for f in glob.glob(os.path.join(C.FEATURES_ACC_DIR, 'features_ACC_EEG_SD_*_mov.csv')))
if a.patients:
    runs = [r for r in runs if r.split('_')[0] in a.patients]


def traiter(run):
    sortie = os.path.join(C.GRILLE_DIR, f'{run}.parquet')
    if os.path.exists(sortie):
        return 'déjà fait'
    if not os.path.exists(P.chemin_hrv(run)):
        return 'HRV absente'
    g = P.fusionner_run(run)
    if not len(g):
        return 'aucune seconde HRV + ACC'
    g.to_parquet(sortie, index=False)
    return 'ok'


res = en_parallele(traiter, runs, a.jobs, 'fusion')
for k in sorted(set(res)):
    log.info(f'{k} : {res.count(k)} runs')
