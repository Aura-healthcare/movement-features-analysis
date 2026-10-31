"""Modèle (version 1, gelée) : recette d'entraînement personnalisé, logique d'alarme et seuils, figés après les phases
de développement. Toute évaluation de référence (lockbox, validation externe) utilise CE module, sans
modification ; un changement = une nouvelle version, réévaluée sur de nouvelles données.

Le « modèle » d'un dispositif personnalisé est une RECETTE appliquée à chaque patient :
1. features `hrv_norme_custom` (59) calculées seconde par seconde (HRV + norme de l'accélération + 4 bandes custom) ;
2. gradient boosting HistGradientBoosting (50 arbres, profondeur 4, taux 0,1), entraîné sur les données du patient,
   toutes ses crises (tous types) en positif, sous-échantillonnage des secondes sans crise à 2 % de crises ;
3. logique d'alarme : moyenne glissante causale sur 10 s (par segment continu), seuil global, fusion des alarmes
   séparées de moins de 60 s, pas de période réfractaire, probabilités brutes (pas de calibration) ;
4. seuil global de production choisi en développement : 0,818 pour ≤ 2 FA / 24 h (point de fonctionnement
   principal), 0,924 / 0,777 / 0,622 pour ≤ 1 / 3 / 5 FA / 24 h.

Évaluation d'un patient non vu : blocs temporels (5 blocs, crises à < 10 min jamais séparées) ; pour chaque bloc,
modèle entraîné sur les 4 autres blocs du patient. Si ces 4 blocs ne contiennent aucune crise, le modèle ne peut pas
être personnalisé : probabilité 0 sur le bloc (aucune alarme), règle fixée avant l'évaluation.
"""
import hashlib
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config as C
from . import decoupages as DC
from . import donnees as D
from . import scoring as S

VERSION = '1.0'
JEU_FEATURES = 'hrv_norme_custom'
HYPERPARAMETRES = {'max_iter': 50, 'learning_rate': 0.1, 'max_depth': 4}
PART_CRISES = 0.02
GRAINE = 42
POST_TRAITEMENT = {'method': 'moyenne', 'n': 10}
SEUILS = {1: 0.92414, 2: 0.81757, 3: 0.7773, 5: 0.62246}      # FA / 24 h visées -> seuil (07.05, règle standard)
SEUIL_PRINCIPAL = SEUILS[2]
FICHIERS_CODE = ['config.py', 'donnees.py', 'decoupages.py', 'scoring.py', 'alarme.py', 'rapport.py', 'modele.py']


def empreinte_code():
    """SHA-256 de chaque fichier de lib/ : trace la version exacte du code évalué."""
    d = os.path.dirname(os.path.abspath(__file__))
    return {f: hashlib.sha256(open(os.path.join(d, f), 'rb').read()).hexdigest() for f in FICHIERS_CODE}


def modele():
    return HistGradientBoostingClassifier(early_stopping=False, random_state=GRAINE, **HYPERPARAMETRES)


def matrice(df):
    feats = C.feature_set(df.columns, JEU_FEATURES)
    if len(feats) != 59:
        raise ValueError(f'{len(feats)} features au lieu de 59')
    return df[feats].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32), feats


def blocs(df):
    """Découpage en 5 blocs temporels (lib/decoupages, version 2) : (bloc de chaque seconde, description)."""
    sz = D.seizures(df)
    groups = D.seizure_groups(sz)
    fold_of, extra = DC.blocs_temporels(df, groups)
    return fold_of, DC.describe(df, fold_of, sz, {**extra, 'groupes': groups})


def predictions_personnalisees(df, fold_of):
    """Probabilité hors échantillon de chaque seconde : pour chaque bloc, modèle entraîné sur les autres blocs."""
    X, _ = matrice(df)
    y = df['label'].values.astype(np.int8)
    proba = np.zeros(len(df), np.float32)
    infos = []
    for f in np.unique(fold_of):
        tr = np.flatnonzero(fold_of != f)
        te = fold_of == f
        if y[tr].sum() == 0:
            infos.append({'bloc': int(f), 'entraine': False, 'crises_train_s': 0})
            continue
        idx = tr[DC.undersample(y[tr], PART_CRISES, GRAINE + int(f))]
        proba[te] = modele().fit(X[idx], y[idx]).predict_proba(X[te])[:, 1]
        infos.append({'bloc': int(f), 'entraine': True, 'crises_train_s': int(y[tr].sum()), 'lignes_train': len(idx)})
    return proba, infos


def entrainer_production(df):
    """Modèle de production d'un patient : entraîné sur TOUTES ses données (après la phase de personnalisation)."""
    X, feats = matrice(df)
    y = df['label'].values.astype(np.int8)
    idx = DC.undersample(y, PART_CRISES, GRAINE)
    return modele().fit(X[idx], y[idx]), feats


def alarmes(proba, segment, seuil=SEUIL_PRINCIPAL):
    """Alarmes [début, fin) en positions de secondes, avec la logique gelée."""
    score = S.postprocess(proba, segment, **POST_TRAITEMENT)
    return S.alarms(score, segment, seuil)
