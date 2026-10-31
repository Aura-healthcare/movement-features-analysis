"""Étape 9 : appliquer le modèle d'un patient à un nouvel enregistrement et produire les alarmes.

Entrée : un parquet au format de 05_patients (mêmes colonnes, préparé par les étapes 1 à 5 sur les nouvelles données).
Sortie : CSV des alarmes (début, fin, durée) et CSV des probabilités seconde par seconde.
Exemple : python 09_surveiller.py --patient sub-001 --donnees nouveau/sub-001.parquet --sortie alarmes_sub-001
"""
import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sizeit import config as C, donnees as D, modele as M, scoring as S  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--patient', required=True)
p.add_argument('--donnees', required=True, help='parquet du nouvel enregistrement')
p.add_argument('--sortie', required=True, help='préfixe des fichiers de sortie')
p.add_argument('--fa', type=int, default=2, choices=sorted(M.SEUILS), help='fausses alarmes / 24 h visées (choix du seuil)')
a = p.parse_args()

modele = joblib.load(os.path.join(C.MODELES_DIR, f'{a.patient}.joblib'))
meta = json.load(open(os.path.join(C.MODELES_DIR, f'{a.patient}.json')))
C.DATA_DIR = os.path.dirname(os.path.abspath(a.donnees))
df = D.load_patient(os.path.basename(a.donnees)[:-len('.parquet')])
X, feats = M.matrice(df)
assert feats == meta['features'], 'les colonnes ne correspondent pas à celles du modèle'
proba = modele.predict_proba(X)[:, 1]
seuil = M.SEUILS[a.fa]
score = S.postprocess(proba, df['segment'].values, **M.POST_TRAITEMENT)
ev = S.alarms(score, df['segment'].values, seuil)
al = pd.DataFrame({'run_id': df['run_id'].astype(str).values[ev[:, 0]], 'debut_elapsed_s': df['elapsed_s'].values[ev[:, 0]],
                   'fin_elapsed_s': df['elapsed_s'].values[ev[:, 1] - 1], 'duree_s': ev[:, 1] - ev[:, 0]}) if len(ev) else pd.DataFrame()
al.to_csv(f'{a.sortie}_alarmes.csv', index=False)
pd.DataFrame({'run_id': df['run_id'].astype(str), 'elapsed_s': df['elapsed_s'], 'proba': proba, 'proba_lissee': score,
              'alarme': np.isin(np.arange(len(df)), np.concatenate([np.arange(s, e) for s, e in ev]) if len(ev) else [])}) \
    .to_parquet(f'{a.sortie}_probabilites.parquet', index=False)
print(f'{len(ev)} alarmes (seuil {seuil:.3f}, ≈ {a.fa} FA / 24 h visées) sur {len(df) / 3600:.1f} h -> {a.sortie}_alarmes.csv')
