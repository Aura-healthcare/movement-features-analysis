"""Étape 3 : features de l'accélération (fenêtres de 10 s, pas de 5 s) -> 03_features_acc. Étape la plus longue."""
import glob
import os
from _commun import arguments, en_parallele, log
from sizeit import config as C, features_acc as F

a = arguments(__doc__)
os.makedirs(C.FEATURES_ACC_DIR, exist_ok=True)
csv = sorted(glob.glob(os.path.join(C.CSV_ACC_RENORM_DIR, '*_mov.csv')))
if a.patients:
    csv = [f for f in csv if os.path.basename(f).split('_')[0] in a.patients]


def traiter(f):
    sortie = os.path.join(C.FEATURES_ACC_DIR, f'features_ACC_EEG_SD_{os.path.basename(f)}')
    if os.path.exists(sortie):
        return -1
    n = F.calculer_fichier(f, sortie + '.tmp')
    if n:
        os.replace(sortie + '.tmp', sortie)
    return n


res = en_parallele(traiter, csv, a.jobs, 'features ACC')
log.info(f'{sum(r > 0 for r in res)} fichiers calculés, {sum(r == -1 for r in res)} déjà présents, {sum(r == 0 for r in res)} en échec')
