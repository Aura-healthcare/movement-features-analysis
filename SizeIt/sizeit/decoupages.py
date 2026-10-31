"""Découpages train / test (version 2, ordre des runs corrigé).

Trois découpages, tous définis en positions dans l'ordre de `donnees.load_patient` :
- `blocs_temporels` : 5 blocs contigus par patient (algorithme de decoupage_train_test.ipynb : frontières hors des
  groupes de crises, crises réparties dans le plus de blocs possible, puis rééquilibrage par déplacement de groupes
  entiers). Évalue un modèle personnalisé sur des périodes qu'il n'a pas vues.
- `un_groupe_de_crises` (leave-one-seizure-group-out) : un fold par groupe de crises (crises à < 10 min) ; le
  fold contient le groupe et son « territoire » (jusqu'à mi-distance des groupes voisins). Répond à « le modèle
  du patient détecte-t-il sa prochaine crise ? ».
- `un_patient` (leave-one-patient-out) : niveau patient, pour le point de départ avant personnalisation et
  l'étude des profils.

Les jeux de test d'un patient recouvrent toujours exactement tout son enregistrement.
"""
import json
import numpy as np
import pandas as pd

from . import config as C
from . import donnees as D

# paramètres des blocs temporels (identiques à decoupage_train_test.ipynb)
N_FOLDS = 5
CUT_MARGIN_S = 60
MIN_FOLD_FRAC = 0.5
MIN_FOLDS_WITH_SEIZURE = 3
REBALANCE_TRIGGER_SHARE = 0.6
CONTEXT_S = 1800
GRID_MAX_POINTS = 3000


# ---------------------------------------------------------------- blocs temporels
def _candidate_cuts(df, groups, n):
    step = max(60, int(np.ceil(n / GRID_MAX_POINTS)))
    cand = set(range(step, n, step))
    cand.update((np.flatnonzero(df['rn'].values[1:] != df['rn'].values[:-1]) + 1).tolist())
    cand = np.array(sorted(cand))
    ok = np.ones(len(cand), dtype=bool)
    for g in groups:
        ok &= ~((cand > g['start'] - CUT_MARGIN_S) & (cand <= g['end'] + CUT_MARGIN_S))
    return cand[ok]


def _best_split(n, groups, cuts, k=N_FOLDS, min_frac=MIN_FOLD_FRAC):
    nodes = np.r_[0, cuts, n].astype(np.int64)
    m = len(nodes)
    starts = np.array([g['start'] for g in groups])
    cnt = np.searchsorted(starts, nodes, side='left')
    target = n / k
    L = nodes[None, :] - nodes[:, None]
    step_cost = np.where(L >= min_frac * target, ((L - target) / target) ** 2, np.inf).astype(np.float32)
    has = (cnt[None, :] - cnt[:, None]) > 0
    cost_no = np.where(~has, step_cost, np.inf).astype(np.float32)
    cost_yes = np.where(has, step_cost, np.inf).astype(np.float32)
    del L, step_cost, has
    INF = np.float32(np.inf)
    dp = np.full((k + 1, k + 1, m), INF, dtype=np.float32)
    back = np.full((k + 1, k + 1, m, 2), -1, dtype=np.int32)
    dp[0, 0, 0] = 0
    for b in range(1, k + 1):
        for s in range(0, b + 1):
            best = np.full(m, INF, dtype=np.float32); arg = np.full(m, -1); prev = np.full(m, -1)
            for cost, ps in ((cost_no, s), (cost_yes, s - 1)):
                if ps < 0 or not np.isfinite(dp[b - 1, ps]).any():
                    continue
                tot = dp[b - 1, ps][:, None] + cost
                i = tot.argmin(axis=0); v = tot[i, np.arange(m)]
                upd = v < best; best[upd], arg[upd], prev[upd] = v[upd], i[upd], ps
            dp[b, s] = best; back[b, s, :, 0] = arg; back[b, s, :, 1] = prev
    feasible = [s for s in range(k + 1) if np.isfinite(dp[k, s, m - 1])]
    if not feasible:
        return None
    j, s, bounds = m - 1, max(feasible), [n]
    for b in range(k, 0, -1):
        i, ps = back[b, s, j]
        bounds.append(int(nodes[i])); j, s = i, ps
    return sorted(bounds)


def _rebalance(fold_of, groups, n, k=N_FOLDS):
    starts = [g['start'] for g in groups]; ends = [g['end'] for g in groups]
    gf = [int(fold_of[a]) for a in starts]
    counts = np.bincount(gf, minlength=k)
    if counts.max() / len(groups) <= REBALANCE_TRIGGER_SHARE and (counts > 0).sum() >= min(MIN_FOLDS_WITH_SEIZURE, len(groups)):
        return []
    moves, moved = [], set()
    while True:
        counts = np.bincount(gf, minlength=k)
        if counts.max() - counts.min() <= 1:
            break
        src, dst = int(counts.argmax()), int(np.flatnonzero(counts == counts.min())[0])
        cand = [i for i in range(len(groups)) if gf[i] == src and i not in moved]
        if not cand:
            break
        iso = lambda i: min(starts[i] - (ends[i - 1] if i else 0), (starts[i + 1] if i + 1 < len(groups) else n) - ends[i])
        i = max(cand, key=iso)
        lo = max(starts[i] - CONTEXT_S, (ends[i - 1] + starts[i]) // 2 if i else 0)
        hi = min(ends[i] + CONTEXT_S, (ends[i] + starts[i + 1]) // 2 if i + 1 < len(groups) else n)
        seg = fold_of[lo:hi]; seg[seg == src] = dst
        gf[i] = dst; moved.add(i)
        moves.append({'seizure_ids': groups[i]['seizure_ids'], 'de': src, 'vers': dst, 'start': int(lo), 'end': int(hi)})
    return moves


def blocs_temporels(df, groups):
    n = len(df)
    cuts = _candidate_cuts(df, groups, n)
    bounds = _best_split(n, groups, cuts)
    fold_of = np.empty(n, dtype=np.int16)
    for f in range(N_FOLDS):
        fold_of[bounds[f]:bounds[f + 1]] = f
    moves = _rebalance(fold_of, groups, n)
    return fold_of, {'deplacements': moves}


# ---------------------------------------------------------------- un groupe de crises par fold
def un_groupe_de_crises(df, groups):
    """Fold i = groupe i + territoire jusqu'à mi-distance des groupes voisins (bornes au milieu des intervalles)."""
    n = len(df)
    bounds = [0] + [(groups[i]['end'] + groups[i + 1]['start']) // 2 for i in range(len(groups) - 1)] + [n]
    fold_of = np.empty(n, dtype=np.int16)
    for i in range(len(groups)):
        fold_of[bounds[i]:bounds[i + 1]] = i
    return fold_of, {}


# ---------------------------------------------------------------- sérialisation
def _segments(fold_of, f):
    m = np.diff(np.r_[False, fold_of == f, False].astype(np.int8))
    return list(zip(np.flatnonzero(m == 1), np.flatnonzero(m == -1)))


def describe(df, fold_of, sz, extra=None):
    folds = []
    for f in sorted(np.unique(fold_of)):
        segs = [{'start': int(a), 'end': int(b),
                 'premiere_ligne': [int(df.at[a, 'rn']), float(df.at[a, 'elapsed_s'])],
                 'derniere_ligne': [int(df.at[b - 1, 'rn']), float(df.at[b - 1, 'elapsed_s'])]}
                for a, b in _segments(fold_of, f)]
        in_f = sz[fold_of[sz['start'].values] == f]
        n_rows = int((fold_of == f).sum())
        folds.append({'fold': int(f), 'segments': segs, 'heures_test': round(n_rows / 3600, 3),
                      'part_test': round(n_rows / len(df), 4), 'crises_test': in_f['seizure_id'].tolist(),
                      'groupes_test': sorted(set(in_f['groupe'].dropna()))})
    assert (fold_of[sz['start'].values] == fold_of[sz['end'].values - 1]).all(), 'une crise est coupée'
    return {'n_lignes': len(df), 'folds': folds, **(extra or {})}


def assign_folds(df, entry):
    """Tableau (len(df),) du fold de test de chaque ligne ; vérifie les bornes enregistrées."""
    if len(df) != entry['n_lignes']:
        raise ValueError('nombre de lignes différent du découpage enregistré')
    fold_of = np.full(len(df), -1, dtype=np.int16)
    for f in entry['folds']:
        for s in f['segments']:
            a, b = s['start'], s['end']
            for row, ref in ((a, s['premiere_ligne']), (b - 1, s['derniere_ligne'])):
                if int(df.at[row, 'rn']) != ref[0] or float(df.at[row, 'elapsed_s']) != ref[1]:
                    raise ValueError(f'borne {row} incohérente avec le découpage enregistré')
            fold_of[a:b] = f['fold']
    if (fold_of < 0).any():
        raise ValueError('des lignes ne sont dans aucun fold')
    return fold_of


def save(splits, meta, path):
    with open(path, 'w') as fh:
        json.dump({'meta': meta, 'patients': splits}, fh, indent=1, ensure_ascii=False)


def load(path):
    with open(path) as fh:
        return json.load(fh)


def undersample(y, target_pos_frac, seed):
    """Indices à garder dans un train : toutes les crises + juste assez de label 0 pour target_pos_frac."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_keep = min(len(neg), int(round(len(pos) * (1 - target_pos_frac) / target_pos_frac)))
    return np.sort(np.r_[pos, rng.choice(neg, n_keep, replace=False)])
