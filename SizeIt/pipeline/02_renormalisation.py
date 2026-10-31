"""Étape 2 : renormalisation de l'accélération (norme moyenne de chaque fichier ramenée à 1) -> 02_csv_acc_renormalise."""
import glob
import os
import pandas as pd
from _commun import arguments, en_parallele, log
from sizeit import config as C, preparation as P

a = arguments(__doc__)
os.makedirs(C.CSV_ACC_RENORM_DIR, exist_ok=True)
csv = sorted(glob.glob(os.path.join(C.CSV_ACC_DIR, '*_mov.csv')))
if a.patients:
    csv = [f for f in csv if os.path.basename(f).split('_')[0] in a.patients]
res = en_parallele(P.renormaliser_fichier, csv, a.jobs, 'renormalisation')
pd.DataFrame([{'fichier': os.path.basename(s), 'facteur_eeg_sd': f1, 'facteur_ecgemg_sd': f2} for s, f1, f2 in res if s]) \
    .to_csv(os.path.join(C.WORK_DIR, 'facteurs_renormalisation.csv'), index=False, mode='a',
            header=not os.path.exists(os.path.join(C.WORK_DIR, 'facteurs_renormalisation.csv')))
log.info(f'{sum(s is not None for s, _, _ in res)} fichiers renormalisés, {sum(s is None for s, _, _ in res)} sans accéléromètre EEG SD (ignorés)')
