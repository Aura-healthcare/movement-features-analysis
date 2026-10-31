"""Accès aux données pour l'outil de visualisation : catalogue des modèles, prédictions, données patient,
crises, score post-traité et alarmes (même règle que sizeit/scoring.py), accélération brute à 25 Hz."""
import os
import sys
import glob
from functools import lru_cache

import numpy as np
import pandas as pd

import config_visu as CV
from sizeit import config as C, donnees as D, scoring as S   # noqa: E402

GROUP_SLUGS = {'hypermotrices_convulsives': 'Hypermotrices et convulsives',
               'motrices_focales_discretes': 'Motrices focales discrètes',
               'non_motrices': 'Non motrices', 'semiologie_non_precisee': 'Sémiologie non précisée'}


# ---------------------------------------------------------------- catalogue des modèles
@lru_cache(maxsize=1)
def catalogue():
    """{nom lisible: {'fichier', 'groupe'}} : chaque parquet de prédictions trouvé sous le dossier des résultats
    (colonnes patient_id, run_id, elapsed_s, proba ; ex. 07_resultats/predictions.parquet)."""
    import pyarrow.parquet as pq
    out = {}
    for f in sorted(glob.glob(os.path.join(CV.RESULTATS_DIR, '**', '*.parquet'), recursive=True)):
        cols = set(pq.read_schema(f).names)
        if {'patient_id', 'run_id', 'elapsed_s', 'proba'} <= cols:
            nom = os.path.relpath(f, CV.RESULTATS_DIR)[:-len('.parquet')].replace(os.sep, ' / ')
            slug = os.path.basename(os.path.dirname(f))
            out[nom] = {'fichier': f, 'groupe': GROUP_SLUGS.get(slug)}
    if not out:
        raise SystemExit(f'aucune prédiction trouvée dans {CV.RESULTATS_DIR} : lancer d’abord pipeline/07_evaluation.py')
    return out


@lru_cache(maxsize=8)
def _preds(fichier):
    return pd.read_parquet(fichier)


@lru_cache(maxsize=64)
def patients_du_modele(model):
    return sorted(_preds(catalogue()[model]['fichier'])['patient_id'].astype(str).unique())


@lru_cache(maxsize=16)
def predictions(model, pid):
    """Prédictions du patient : run_id, elapsed_s, proba (+ crf_proba / crf_state si présents)."""
    p = _preds(catalogue()[model]['fichier'])
    p = p[p['patient_id'].astype(str) == pid]
    keep = [c for c in ['run_id', 'elapsed_s', 'proba', 'crf_proba', 'crf_state'] if c in p.columns]
    return p[keep].astype({'run_id': str})


# ---------------------------------------------------------------- données patient + prédictions alignées
@lru_cache(maxsize=16)
def patient(model, pid):
    """DataFrame dans l'ordre corrigé (lib/donnees) avec FC, agitation et les probabilités du modèle
    (NaN là où le modèle n'a pas de prédiction, ex. crises d'un autre groupe pour un modèle par groupe)."""
    df = D.load_patient(pid, columns=[CV.HR_COL, CV.ACT_COL] + features_disponibles(pid))
    p = predictions(model, pid)
    m = df[['run_id', 'elapsed_s']].astype({'run_id': str}).merge(p, on=['run_id', 'elapsed_s'], how='left')
    for c in [c for c in p.columns if c not in ('run_id', 'elapsed_s')]:
        df[c] = m[c].values
    df['heure_cumulee'] = np.arange(len(df)) / 3600
    return df


@lru_cache(maxsize=64)
def features_disponibles(pid):
    import pyarrow.parquet as pq
    cols = set(pq.read_schema(os.path.join(CV.DATA_DIR, f'{pid}.parquet')).names)
    return [c for c in CV.EXTRA_FEATURES if c in cols]


@lru_cache(maxsize=16)
def crises(pid):
    return D.seizures(D.load_patient(pid, columns=[]))


@lru_cache(maxsize=16)
def bornes_segments(pid):
    return D.segment_bounds(D.load_patient(pid, columns=[]))


def evaluer(model, pid, signal='proba', methode='moyenne', n=10, k=5, seuil=0.5, regle='standard'):
    """Score post-traité, alarmes et statut de chaque crise / alarme (lib/scoring)."""
    df = patient(model, pid)
    proba = np.nan_to_num(df[signal].values.astype(float), nan=0.0)
    score = S.postprocess(proba, df['segment'].values, method=methode, n=n, k=k)
    ev = S.alarms(score, df['segment'].values, seuil)
    grp = catalogue()[model]['groupe']
    sz, al = S.score_patient(ev, crises(pid), bornes_segments(pid), rule=regle,
                             in_scope=[grp] if grp else None)
    return score, sz, al


# ---------------------------------------------------------------- accélération brute
@lru_cache(maxsize=64)
def _raw_window(run_id, e0, e1):
    path = os.path.join(CV.RAW_ACC_DIR, f'{run_id}_mov.csv')
    if not os.path.exists(path):
        return pd.DataFrame(columns=['t_run_s'] + CV.RAW_COLS)
    a, n = int(e0 * CV.RAW_FS), int((e1 - e0 + 1) * CV.RAW_FS)
    r = pd.read_csv(path, skiprows=range(1, a + 1), nrows=n, usecols=CV.RAW_COLS)
    r.insert(0, 't_run_s', e0 + np.arange(len(r)) / CV.RAW_FS)
    return r


def accel_brute(df, pos_debut, pos_fin, pos_ref):
    """Accélération brute (25 Hz) entre deux positions de df ; temps en secondes relatif à pos_ref.
    Gère les fenêtres qui traversent plusieurs runs recollés."""
    w = df.iloc[pos_debut:pos_fin]
    out = []
    for run_id, g in w.groupby('run_id', sort=False, observed=True):
        e0, e1 = float(g['elapsed_s'].min()), float(g['elapsed_s'].max())
        r = _raw_window(str(run_id), e0, e1).copy()
        if not len(r):
            continue
        # position de la seconde e0 de ce run dans df, puis temps relatif au repère
        p0 = int(g.index.min())
        r['t'] = (p0 - pos_ref) + (r['t_run_s'] - e0)
        out.append(r)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=['t'] + CV.RAW_COLS)
