"""Préparation des données : EDF -> CSV d'accélération, renormalisation, fusion HRV + ACC + annotations en une grille
à 1 s par run, regroupement par patient avec identifiant de crise (seizure_id).

Repris des notebooks d'origine (SZIT_Edf_to_csv_acc, SZIT_Renormalisation_acc, 01_data_preparation,
build_seizure_id) ; la fusion reproduit à l'identique la grille « feat-grid-intersect » utilisée pour le modèle.
"""
import glob
import os
import re

import numpy as np
import pandas as pd

from . import config as C

AXES_EEG = ['EEG SD ACC X', 'EEG SD ACC Y', 'EEG SD ACC Z']
AXES_ECGEMG = ['ECGEMG SD ACC X', 'ECGEMG SD ACC Y', 'ECGEMG SD ACC Z']
TOLERANCE_ACC_S = 2.5          # une ligne de features ACC (toutes les 5 s) est rattachée aux secondes à ± 2,5 s


def nom_run(chemin):
    """'.../sub-001_ses-01_task-szMonitoring_run-01_mov.edf' -> 'sub-001_ses-01_task-szMonitoring_run-01'."""
    return re.sub(r'_(mov|ecg|eeg)\.(edf|csv)$', '', os.path.basename(chemin))


def runs_mov(raw_dir=C.RAW_DIR, patients=None):
    fichiers = sorted(glob.glob(os.path.join(raw_dir, 'sub-*', 'ses-*', 'mov', '*_mov.edf')))
    if patients:
        fichiers = [f for f in fichiers if os.path.basename(f).split('_')[0] in patients]
    return fichiers


# ---------------------------------------------------------------- 1. EDF -> CSV
def edf_vers_csv(chemin_edf, dossier_sortie=C.CSV_ACC_DIR):
    """Canaux d'accélération d'un EDF « mov » vers un CSV horodaté (5 décimales, comme les fichiers d'origine)."""
    import pyedflib
    sortie = os.path.join(dossier_sortie, os.path.basename(chemin_edf).replace('.edf', '.csv'))
    if os.path.exists(sortie):
        return sortie
    f = pyedflib.EdfReader(chemin_edf)
    try:
        labels = f.getSignalLabels()
        idx = [i for i, lab in enumerate(labels) if 'ACC' in lab.upper()]
        if not idx:
            return None
        df = pd.DataFrame({labels[i]: pd.Series(f.readSignal(i)) for i in idx})
        fs = f.getSampleFrequency(idx[0])
        df.insert(0, 'time', pd.Timestamp(f.getStartdatetime()) + pd.to_timedelta(np.arange(len(df)) / fs, unit='s'))
    finally:
        f.close()
    df.to_csv(sortie + '.tmp', index=False, float_format='%.5f', date_format='%Y-%m-%d %H:%M:%S.%f')
    os.replace(sortie + '.tmp', sortie)
    return sortie


# ---------------------------------------------------------------- 2. renormalisation
def renormaliser(df, axes, norme):
    """Divise les axes par la norme moyenne du fichier (norme moyenne ramenée à 1, ≈ 1 g) et recalcule la norme."""
    n = np.linalg.norm(df[axes].values, axis=1)
    facteur = n.mean()
    df[axes] *= (1 / facteur)
    df[norme] = np.linalg.norm(df[axes].values, axis=1)
    return df, facteur


def renormaliser_fichier(chemin_csv, dossier_sortie=C.CSV_ACC_RENORM_DIR):
    sortie = os.path.join(dossier_sortie, os.path.basename(chemin_csv))
    if os.path.exists(sortie):
        return sortie, np.nan, np.nan
    df = pd.read_csv(chemin_csv)
    if not set(AXES_EEG) <= set(df.columns):
        return None, np.nan, np.nan                       # pas d'accéléromètre EEG SD : run ignoré (comme à l'origine)
    df, f1 = renormaliser(df, AXES_EEG, 'norme_1')
    f2 = np.nan
    if set(AXES_ECGEMG) <= set(df.columns):
        df, f2 = renormaliser(df, AXES_ECGEMG, 'norme_2')
    df.to_csv(sortie + '.tmp')
    os.replace(sortie + '.tmp', sortie)
    return sortie, f1, f2


# ---------------------------------------------------------------- 4. fusion HRV + ACC + annotations
def chemin_hrv(run, hrv_dir=C.HRV_DIR):
    return os.path.join(hrv_dir, f'{run}_ecg', 'fast', 'features', f'feats_{run}_ecg_fast.csv')


def chemin_events(run, raw_dir=C.RAW_DIR):
    pid = run.split('_')[0]
    return os.path.join(raw_dir, pid, 'ses-01', 'eeg', f'{run}_events.tsv')


def fusionner_run(run, features_acc_dir=C.FEATURES_ACC_DIR, hrv_dir=C.HRV_DIR, raw_dir=C.RAW_DIR):
    """Grille à 1 s d'un run : une ligne par seconde de HRV, features ACC de la ligne la plus proche (± 2,5 s, en temps
    écoulé depuis le début du run), label = 1 pendant les crises annotées (onset <= t <= onset + durée). Seules les
    secondes où HRV ET accélération sont disponibles sont gardées (« intersect »)."""
    pid = run.split('_')[0]
    h = pd.read_csv(chemin_hrv(run, hrv_dir)).drop(columns=['filename', 'original_filename'], errors='ignore')
    h.columns = ['hrv_' + c if c != 'timestamp' else c for c in h.columns]
    h['elapsed_s'] = h['hrv_interval_start_time'] / 1000
    a = pd.read_csv(os.path.join(features_acc_dir, f'features_ACC_EEG_SD_{run}_mov.csv'))
    a = a.drop(columns=[c for c in a.columns if c.startswith('Unnamed')])
    t = pd.to_datetime(a['time'])
    a['_t'] = (t - t.iloc[0]).dt.total_seconds()
    a = a.drop(columns=['time'])
    a.columns = ['acc_' + c if c != '_t' else c for c in a.columns]
    g = pd.merge_asof(h.sort_values('elapsed_s'), a.sort_values('_t'), left_on='elapsed_s', right_on='_t',
                      direction='nearest', tolerance=TOLERANCE_ACC_S)
    g['hrv_available'] = True
    g['acc_available'] = g['_t'].notna()
    g = g.drop(columns=['_t'])
    ev = pd.read_csv(chemin_events(run, raw_dir), sep='\t')
    g['label'] = 0
    g['seizure_type'] = pd.Series(np.nan, index=g.index, dtype=object)
    for r in ev[ev['eventType'].astype(str).str.startswith('sz')].itertuples():
        m = (g['elapsed_s'] >= r.onset) & (g['elapsed_s'] <= r.onset + r.duration)
        g.loc[m, 'label'] = 1
        g.loc[m, 'seizure_type'] = r.eventType
    g['patient_id'] = pid
    g['run_id'] = run
    cols = (['elapsed_s'] + [c for c in g.columns if c.startswith('hrv_') and c != 'hrv_available'] + ['hrv_available']
            + [c for c in g.columns if c.startswith('acc_') and c != 'acc_available'] + ['acc_available']
            + ['label', 'seizure_type', 'timestamp', 'patient_id', 'run_id'])
    return g.loc[g['acc_available'], cols].reset_index(drop=True)


# ---------------------------------------------------------------- 5. un fichier par patient, identifiant de crise
def ajouter_seizure_id(df, patient_id):
    """Crises = blocs contigus de label 1 ; les runs sont ordonnés par leur premier horodatage ; une crise à cheval sur
    deux runs consécutifs (positive de part et d'autre de la frontière) garde un seul identifiant <patient>_<n>."""
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    ordre = df.groupby('run_id', observed=True)['timestamp'].min().sort_values()
    rang = {r: i for i, r in enumerate(ordre.index)}
    df['_rang'] = df['run_id'].map(rang)
    df = df.sort_values(['_rang', 'elapsed_s']).drop(columns='_rang').reset_index(drop=True)
    pos = df['label'].values == 1
    run = df['run_id'].values
    change_run = np.r_[True, run[1:] != run[:-1]]
    change_label = np.r_[True, pos[1:] != pos[:-1]]
    positif_aux_bords = np.r_[False, pos[1:] & pos[:-1]]
    bloc = np.cumsum(change_label | (change_run & ~positif_aux_bords))
    sid = np.full(len(df), None, dtype=object)
    for num, b in enumerate(pd.unique(bloc[pos]), start=1):
        sid[pos & (bloc == b)] = f'{patient_id}_{num}'
    df['seizure_id'] = sid
    return df


def preparer_patient(pid, grille_dir=C.GRILLE_DIR, sortie_dir=C.DATA_DIR):
    """Concatène les grilles des runs d'un patient, features en float32, ajoute seizure_id, écrit <pid>.parquet."""
    fichiers = sorted(glob.glob(os.path.join(grille_dir, f'{pid}_*.parquet')))
    if not fichiers:
        return None
    df = pd.concat([pd.read_parquet(f) for f in fichiers], ignore_index=True)
    feats = [c for c in df.columns if c not in C.META_COLS]
    df[feats] = df[feats].astype('float32')
    for c in ('hrv_interval_index', 'hrv_available', 'acc_available', 'label'):
        df[c] = df[c].astype('float32')
    df = ajouter_seizure_id(df, pid)
    for c in ('patient_id', 'run_id', 'seizure_type'):
        df[c] = df[c].astype('category')
    sortie = os.path.join(sortie_dir, f'{pid}.parquet')
    df.to_parquet(sortie + '.tmp', index=False)
    os.replace(sortie + '.tmp', sortie)
    return sortie
