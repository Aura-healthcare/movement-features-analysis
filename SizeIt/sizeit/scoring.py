"""Comptage des détections et des fausses alarmes, règle unique pour tous les modèles.

Chaîne : probabilités par seconde → post-traitement (score) → seuil → alarmes → comparaison aux crises.

Post-traitements disponibles (`postprocess`) :
- 'brut'      : la probabilité elle-même ;
- 'moyenne'   : moyenne des n dernières secondes (ancienne règle, n = 10) ;
- 'kieme_max' : k-ième plus grande probabilité des n dernières secondes (ex. k = 5, n = 20). Le score dépasse le
                seuil si et seulement si au moins k des n dernières secondes le dépassent : c'est la règle
                « k secondes sur n », qui impose une alarme soutenue sans être diluée par les creux.
Toutes les fenêtres sont causales (passé uniquement) et calculées dans un même segment continu.

Règles de comptage (`config.SCORING`) :
- 'standard' : une crise est détectée si une alarme chevauche [début - 30 s, fin + 60 s] ;
- 'strict'   : alarme d'au moins 10 s qui chevauche la crise elle-même (sans tolérance).
Une alarme qui ne chevauche aucune fenêtre de crise est une fausse alarme ; une alarme qui chevauche une crise
hors périmètre (ex. type de crise non ciblé, passé dans `neutral`) n'est ni l'un ni l'autre.
"""
import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------- post-traitement
def postprocess(proba, segment, method='moyenne', n=10, k=5):
    p = pd.Series(np.asarray(proba, dtype=float))
    g = p.groupby(np.asarray(segment), sort=False)
    if method == 'brut':
        return p.values
    if method == 'moyenne':
        return g.transform(lambda s: s.rolling(n, min_periods=1).mean()).values
    if method == 'kieme_max':
        # k-ième plus grande valeur parmi les n dernières = quantile d'ordre (n - k) sur n valeurs triées
        q = (n - k) / (n - 1)
        return g.transform(lambda s: s.rolling(n, min_periods=n).quantile(q, interpolation='lower')).fillna(0).values
    raise ValueError(method)


# ---------------------------------------------------------------- alarmes
def alarms(score, segment, threshold, merge_s=C.ALARM_MERGE_S, min_alarm_s=1, refractory_s=0):
    """Alarmes = intervalles [début, fin) de secondes au-dessus du seuil dans un même segment continu.
    Fusion si écart < merge_s ; alarmes plus courtes que min_alarm_s supprimées ; une alarme qui commence moins de
    refractory_s après le début de l'alarme précédente (même segment) est absorbée."""
    a = np.asarray(score) >= threshold
    seg = np.asarray(segment)
    new = np.r_[True, seg[1:] != seg[:-1]]
    last = np.r_[seg[1:] != seg[:-1], True]
    st = np.flatnonzero(a & (new | ~np.r_[False, a[:-1]]))
    en = np.flatnonzero(a & (last | ~np.r_[a[1:], False])) + 1
    ev = []
    for s, e in zip(st, en):
        if ev and seg[s] == seg[ev[-1][0]] and (s - ev[-1][1] < merge_s or s - ev[-1][0] < refractory_s):
            ev[-1][1] = e
        else:
            ev.append([s, e])
    ev = np.array(ev, dtype=np.int64).reshape(-1, 2)
    return ev[(ev[:, 1] - ev[:, 0]) >= min_alarm_s] if len(ev) else ev


def score_patient(ev, sz, seg_bounds, rule='standard', in_scope=None, last_seizure_end=None):
    """Compare les alarmes `ev` aux crises `sz` (sortie de donnees.seizures) d'un patient.

    in_scope : ensemble des groupes de crises ciblés (None = toutes) ; les autres crises sont neutres.
    Renvoie (crises, alarmes) : une ligne par crise ciblée (détectée, latence) et une par alarme
    (statut 'détection' / 'fausse alarme' / 'neutre', durée, post-ictale ou non).
    """
    r = C.SCORING[rule]
    if r['min_alarm_s'] > 1 and len(ev):
        ev = ev[(ev[:, 1] - ev[:, 0]) >= r['min_alarm_s']]
    sz = sz.copy()
    lo = np.array([max(s - r['pre_s'], seg_bounds[g][0]) for s, g in zip(sz['start'], sz['segment'])], dtype=np.int64)
    hi = np.array([min(e + r['post_s'], seg_bounds[g][1]) for e, g in zip(sz['end'], sz['segment'])], dtype=np.int64)
    target = np.ones(len(sz), bool) if in_scope is None else sz['groupe'].isin(in_scope).values
    status = np.full(len(ev), 'fausse alarme', dtype=object)
    det, lat = [], []
    for i in range(len(sz)):
        hit = (ev[:, 0] < hi[i]) & (ev[:, 1] > lo[i]) if len(ev) else np.zeros(0, bool)
        if target[i]:
            status[hit] = 'détection'
            det.append(bool(hit.any()))
            lat.append(float(ev[hit, 0].min() - sz['start'].iat[i]) if hit.any() else np.nan)
        else:
            status[hit & (status != 'détection')] = 'neutre'
            det.append(np.nan); lat.append(np.nan)
    sz['detectee'] = det; sz['latence_s'] = lat; sz['ciblee'] = target
    al = pd.DataFrame(ev, columns=['start', 'end'])
    al['duree_s'] = al['end'] - al['start']
    al['statut'] = status
    # fausse alarme post-ictale : dans l'heure qui suit la fin d'une crise
    ends = np.sort(sz['end'].values)
    if len(al):
        prev_end = np.array([ends[ends <= s].max() if (ends <= s).any() else -np.inf for s in al['start']])
        al['post_ictale'] = (al['statut'] == 'fausse alarme') & (al['start'] - prev_end <= C.POST_ICTAL_S)
    else:
        al['post_ictale'] = []
    return sz[sz['ciblee']], al


# ---------------------------------------------------------------- métriques et intervalles de confiance
def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n; d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return max(0.0, c - h), min(1.0, c + h)


def summarize(per_patient):
    """per_patient : DataFrame (patient_id, n_crises, n_detectees, n_fa, heures, latences=list)."""
    n, k = per_patient['n_crises'].sum(), per_patient['n_detectees'].sum()
    lat = np.concatenate([np.asarray(x, float) for x in per_patient['latences']]) if len(per_patient) else np.array([])
    pp = per_patient[per_patient['n_crises'] > 0]
    return {'n_patients': len(per_patient), 'n_crises': int(n), 'n_detectees': int(k),
            'sensibilite': k / n if n else np.nan,
            'fa_par_24h': per_patient['n_fa'].sum() / per_patient['heures'].sum() * 24,
            'latence_mediane_s': float(np.nanmedian(lat)) if np.isfinite(lat).any() else np.nan,
            'latence_p90_s': float(np.nanpercentile(lat, 90)) if np.isfinite(lat).any() else np.nan,
            'sensibilite_mediane_patients': float((pp['n_detectees'] / pp['n_crises']).median()) if len(pp) else np.nan,
            'part_patients_au_moins_une': float((pp['n_detectees'] > 0).mean()) if len(pp) else np.nan,
            'part_patients_au_moins_50pct': float((pp['n_detectees'] / pp['n_crises'] >= 0.5).mean()) if len(pp) else np.nan}


def bootstrap_ci(per_patient, n_boot=2000, seed=0, keys=('sensibilite', 'fa_par_24h', 'sensibilite_mediane_patients')):
    """IC 95 % par rééchantillonnage des PATIENTS (les crises d'un patient ne sont pas indépendantes)."""
    rng = np.random.default_rng(seed)
    vals = {k: [] for k in keys}
    idx = np.arange(len(per_patient))
    for _ in range(n_boot):
        s = summarize(per_patient.iloc[rng.choice(idx, len(idx), replace=True)])
        for k in keys:
            vals[k].append(s[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in vals.items()}


def sens_at_fa(curve, fa, fa_col='fa_par_24h', sens_col='sensibilite'):
    """Sensibilité interpolée à un nombre de fausses alarmes / 24 h donné (courbe obtenue en faisant varier le seuil)."""
    g = curve.sort_values(fa_col)
    return float(np.interp(fa, g[fa_col], g[sens_col]))


def bootstrap_diff(per_a, per_b, n_boot=2000, seed=0, key='sensibilite'):
    """IC 95 % de la DIFFÉRENCE A - B d'une métrique, bootstrap APPARIÉ par patient (mêmes patients tirés pour
    les deux modèles). per_a / per_b : tableaux par patient (format de summarize) sur les mêmes patients."""
    a = per_a.set_index('patient_id').sort_index(); b = per_b.set_index('patient_id').loc[a.index]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(a)); diffs = []
    for _ in range(n_boot):
        i = rng.choice(idx, len(idx), replace=True)
        diffs.append(summarize(a.iloc[i].reset_index())[key] - summarize(b.iloc[i].reset_index())[key])
    est = summarize(a.reset_index())[key] - summarize(b.reset_index())[key]
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return float(est), (float(lo), float(hi))
