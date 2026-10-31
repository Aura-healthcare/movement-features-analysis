"""Étape 8 : modèle de production de chaque patient, entraîné sur TOUTES ses données -> 08_modeles/<patient>.joblib
(+ .json : recette, features, seuils). C'est ce modèle qui sert ensuite à surveiller le patient (étape 9)."""
import json
import os
import joblib
from _commun import arguments, en_parallele, log
from sizeit import config as C, donnees as D, modele as M

a = arguments(__doc__)
os.makedirs(C.MODELES_DIR, exist_ok=True)
pids = [p for p in D.list_patients(exclude=C.EXCLUDED_FOR_MODELS) if not a.patients or p in a.patients]


def entrainer(pid):
    df = D.load_patient(pid)
    if len(D.seizures(df)) < C.MIN_CRISES_MODELE:
        return None
    modele, feats = M.entrainer_production(df)
    joblib.dump(modele, os.path.join(C.MODELES_DIR, f'{pid}.joblib'))
    json.dump({'patient_id': pid, 'version': M.VERSION, 'features': feats, 'hyperparametres': M.HYPERPARAMETRES,
               'post_traitement': M.POST_TRAITEMENT, 'seuils_fa_par_24h': M.SEUILS, 'seuil_principal': M.SEUIL_PRINCIPAL,
               'crises_entrainement': int(len(D.seizures(df))), 'heures_entrainement': len(df) / 3600},
              open(os.path.join(C.MODELES_DIR, f'{pid}.json'), 'w'), indent=1, ensure_ascii=False)
    return pid


res = en_parallele(entrainer, pids, a.jobs, 'modèles de production')
log.info(f'{sum(r is not None for r in res)} modèles écrits dans {C.MODELES_DIR}')
