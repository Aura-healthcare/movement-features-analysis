"""Chargement des données dans l'ordre chronologique corrigé.

Corrections par rapport aux notebooks précédents :
- les runs sont triés par NUMÉRO (run-9 < run-10 < run-100) et non plus par chaîne de caractères
  (l'ancien tri plaçait run-100..run-112 entre run-09 et run-10 pour sub-089 et sub-092) ;
- les fichiers d'une heure consécutifs sont recollés en « segments continus » (colonne `segment`) :
  un run de 56 à 62 min suivi du run de numéro suivant est un enregistrement continu découpé en fichiers.
"""
import os
import re
import numpy as np
import pandas as pd

from . import config as C


def run_number(run_id):
    return int(re.search(r'run-(\d+)', str(run_id)).group(1))


def list_patients(exclude=()):
    pids = sorted(os.path.splitext(f)[0] for f in os.listdir(C.DATA_DIR) if f.endswith('.parquet'))
    return [p for p in pids if p not in exclude]


def load_patient(pid, columns=None):
    """DataFrame du patient trié par (numéro de run, elapsed_s), index 0..n-1 (= position `pos`).

    Colonnes ajoutées : `rn` (numéro de run), `segment` (segment continu), `type` (type court),
    `groupe` (groupe de crises). `columns=None` charge toutes les colonnes.
    """
    need = ['run_id', 'elapsed_s', 'label', 'seizure_id', 'seizure_type']
    cols = None if columns is None else list(dict.fromkeys(need + list(columns)))
    df = pd.read_parquet(os.path.join(C.DATA_DIR, f'{pid}.parquet'), columns=cols)
    df['rn'] = df['run_id'].map(run_number).astype(np.int32)
    df = df.sort_values(['rn', 'elapsed_s'], kind='stable').reset_index(drop=True)
    df['label'] = df['label'].fillna(0).astype(np.int8)
    # segments continus
    runs = df.groupby('rn', sort=True)['elapsed_s'].agg(['min', 'max', 'size'])
    dur = runs['max'] - runs['min'] + 1
    glue = {}
    rns = runs.index.tolist()
    for a, b in zip(rns[:-1], rns[1:]):
        glue[b] = (b == a + 1) and (C.HOUR_FILE_MIN_S <= dur[a] <= C.HOUR_FILE_MAX_S)
    seg, s = {}, 0
    for i, r in enumerate(rns):
        if i > 0 and not glue[r]:
            s += 1
        seg[r] = s
    df['segment'] = df['rn'].map(seg).astype(np.int32)
    df['type'] = df['seizure_type'].map(C.short_type)
    df['groupe'] = df['type'].map(C.TYPE_TO_GROUP)
    return df


def seizures(df):
    """Une ligne par crise : début / fin (exclue) en position, run, segment, type, groupe, durée (s)."""
    d = df[df['seizure_id'].notna()]
    pos = d.index.values
    sz = (pd.DataFrame({'seizure_id': d['seizure_id'].values, 'pos': pos, 'rn': d['rn'].values,
                        'run_id': d['run_id'].values, 'segment': d['segment'].values, 'type': d['type'].values,
                        'groupe': d['groupe'].values})
          .groupby('seizure_id').agg(start=('pos', 'min'), end=('pos', 'max'), rn=('rn', 'first'),
                                     run_id=('run_id', 'first'), segment=('segment', 'first'),
                                     type=('type', 'first'), groupe=('groupe', 'first')))
    sz['end'] += 1
    sz['duree_s'] = sz['end'] - sz['start']
    sz['categorie'] = sz['groupe'].map(C.CATEGORIES)
    return sz.sort_values('start').reset_index()


def seizure_groups(sz, gap_s=C.SEIZURE_GROUP_GAP_S):
    """Regroupe les crises d'un même segment continu séparées de moins de `gap_s` secondes.
    Renvoie une liste de dicts {'seizure_ids', 'start', 'end'} (end exclu), triée."""
    groups = []
    for _, r in sz.sort_values('start').iterrows():
        if groups and r['segment'] == groups[-1]['segment'] and r['start'] - groups[-1]['end'] < gap_s:
            groups[-1]['end'] = max(groups[-1]['end'], int(r['end']))
            groups[-1]['seizure_ids'].append(r['seizure_id'])
        else:
            groups.append({'seizure_ids': [r['seizure_id']], 'start': int(r['start']), 'end': int(r['end']),
                           'segment': int(r['segment'])})
    return groups


def segment_bounds(df):
    """Bornes (début, fin exclue) en position de chaque segment continu."""
    g = df.groupby('segment', sort=True).apply(lambda d: (d.index.min(), d.index.max() + 1), include_groups=False)
    return dict(g)
