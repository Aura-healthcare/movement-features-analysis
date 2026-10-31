"""Logique d'alarme : comptage par bloc de test pour une grille de post-traitements, de seuils et de périodes
réfractaires, puis choix des seuils (global, par groupe de crises, par patient) sans regarder le bloc évalué.

Chaîne (lib/scoring) : probabilités → post-traitement → seuil → alarmes (fusion < 60 s) → période réfractaire →
comparaison aux crises (règles standard et stricte).

Période réfractaire (`refractaire`) : après une alarme émise, aucune nouvelle alarme n'est émise pendant r secondes
(comptées depuis le DÉBUT de l'alarme émise, dans un même segment continu). Les alarmes supprimées ne comptent ni
comme détections ni comme fausses alarmes : une crise qui ne chevauche qu'une alarme supprimée est manquée.
(À ne pas confondre avec `scoring.alarms(refractory_s=…)`, qui absorbe l'alarme suivante dans la précédente.)

Tables de comptage (`compter`) : une ligne par (post-traitement, seuil, réfractaire, règle, périmètre, bloc) avec
n_crises, n_detectees, n_fa, n_fa_post_ictales, latences ; les heures par bloc sont dans une table à part.
Crise et alarme sont rattachées au bloc de test où elles commencent.
"""
import os

import numpy as np
import pandas as pd

from . import config as C
from . import scoring as S

# seuils régulièrement espacés en logit : resserrés vers 1, là où se situent les points de fonctionnement
SEUILS = np.round(1 / (1 + np.exp(-np.linspace(-3, 7.5, 43))), 5)
FA_CIBLES = [1, 2, 3, 5]


def refractaire(ev, segment, r_s):
    """Alarmes `ev` (n, 2) après suppression de celles qui débutent moins de r_s s après le début de la dernière
    alarme émise du même segment continu."""
    if r_s <= 0 or len(ev) == 0:
        return ev
    seg = np.asarray(segment)
    keep, last_start, last_seg = [], None, None
    for i, (s, _) in enumerate(ev):
        if last_start is not None and seg[s] == last_seg and s - last_start < r_s:
            continue
        keep.append(i); last_start, last_seg = s, seg[s]
    return ev[keep]


def compter(df, fold_of, proba, sz, segb, postprocs, seuils=SEUILS, refractaires=(0,), perimetres=None):
    """Comptages par bloc de test pour un patient.

    df : sortie de donnees.load_patient (ordre corrigé) ; fold_of : bloc de test de chaque seconde ;
    proba : probabilités hors échantillon alignées sur df ; postprocs : {nom: kwargs de scoring.postprocess} ;
    perimetres : {nom: liste de groupes ciblés ou None}.
    """
    perimetres = perimetres or {'toutes': None}
    seg = df['segment'].values
    blocs = np.unique(fold_of)
    rows = []
    for pp_name, pp in postprocs.items():
        score = S.postprocess(proba, seg, **pp)
        for thr in seuils:
            ev0 = S.alarms(score, seg, thr)
            for r in refractaires:
                ev = refractaire(ev0, seg, r)
                for regle in C.SCORING:
                    for per_name, scope in perimetres.items():
                        s, a = S.score_patient(ev, sz, segb, rule=regle, in_scope=scope)
                        bs = fold_of[s['start'].values] if len(s) else np.zeros(0, int)
                        ba = fold_of[a['start'].values] if len(a) else np.zeros(0, int)
                        det = s['detectee'].values.astype(bool) if len(s) else np.zeros(0, bool)
                        fa = (a['statut'] == 'fausse alarme').values if len(a) else np.zeros(0, bool)
                        post = a['post_ictale'].values.astype(bool) if len(a) else np.zeros(0, bool)
                        for b in blocs:
                            ms, ma = bs == b, ba == b
                            rows.append({'post_traitement': pp_name, 'seuil': float(thr), 'refractaire_s': r,
                                         'regle': regle, 'perimetre': per_name, 'bloc': int(b),
                                         'n_crises': int(ms.sum()), 'n_detectees': int((ms & det).sum()),
                                         'n_fa': int((ma & fa).sum()), 'n_fa_post_ictales': int((ma & fa & post).sum()),
                                         'latences': s.loc[ms & det, 'latence_s'].tolist() if len(s) else []})
    return pd.DataFrame(rows)


def heures_par_bloc(fold_of):
    b, n = np.unique(fold_of, return_counts=True)
    return pd.DataFrame({'bloc': b.astype(int), 'heures': n / 3600})


# ---------------------------------------------------------------- agrégation
CLES = ['post_traitement', 'refractaire_s', 'regle', 'perimetre']


def par_patient(counts, heures, cles=CLES + ['seuil']):
    """Somme des blocs : une ligne par patient et configuration (format de scoring.summarize)."""
    c = counts.merge(heures, on=['patient_id', 'bloc'])
    g = c.groupby(cles + ['patient_id'], sort=False)
    out = g[['n_crises', 'n_detectees', 'n_fa', 'n_fa_post_ictales', 'heures']].sum()
    out['latences'] = g['latences'].agg(lambda s: [x for l in s for x in l])    # listes (ou tableaux relus du parquet)
    return out.reset_index()


def courbes(pp):
    """Courbe sensibilité / FA par configuration (somme sur les patients) à partir de `par_patient`."""
    g = pp.groupby(CLES + ['seuil'], sort=False)
    c = g[['n_crises', 'n_detectees', 'n_fa', 'n_fa_post_ictales', 'heures']].sum().reset_index()
    c['sensibilite'] = c['n_detectees'] / c['n_crises']
    c['fa_par_24h'] = c['n_fa'] / c['heures'] * 24
    c['latence_mediane_s'] = g['latences'].apply(lambda l: float(np.median(np.concatenate([np.asarray(x, float) for x in l])))
                                                 if sum(len(x) for x in l) else np.nan).values
    return c


def table_a_fa(pp, curve, fa):
    """Tableau par patient au seuil global dont les fausses alarmes poolées sont les plus proches de `fa`."""
    thr = float(curve.iloc[(curve['fa_par_24h'] - fa).abs().argmin()]['seuil'])
    return pp[pp['seuil'] == thr], thr


# ---------------------------------------------------------------- choix des seuils
def seuil_pour_fa(tab, fa_cible):
    """Plus petit seuil dont les fausses alarmes / 24 h (tableau agrégé par seuil : n_fa, heures) restent ≤ fa_cible
    pour ce seuil et tous les seuils plus hauts ; à défaut, le seuil le plus haut."""
    t = tab.sort_values('seuil')
    ok = (t['n_fa'] / t['heures'] * 24 <= fa_cible).values
    ok_suffix = np.flip(np.cumprod(np.flip(ok))).astype(bool)          # vrai pour ce seuil et tous les suivants
    return float(t['seuil'].values[ok_suffix][0]) if ok_suffix.any() else float(t['seuil'].max())


# ---------------------------------------------------------------- données d'entrée
def charger(pid, preds_path, col='proba'):
    """df (ordre corrigé), bloc de test de chaque seconde, probabilités alignées, crises, bornes des segments."""
    from . import donnees as D, decoupages as DC
    p = pd.read_parquet(preds_path, filters=[('patient_id', '==', pid)], columns=['run_id', 'elapsed_s', col])
    df = D.load_patient(pid, columns=[])
    m = df[['run_id', 'elapsed_s']].astype({'run_id': str}).merge(p.astype({'run_id': str}), on=['run_id', 'elapsed_s'],
                                                                  how='left', validate='one_to_one')
    if m[col].isna().any():
        raise ValueError(f'{pid} : {int(m[col].isna().sum())} secondes sans prédiction')
    cfg = DC.load(os.path.join(C.DECOUPAGES_RES, 'decoupage_blocs_temporels.json'))
    fold_of = DC.assign_folds(df, cfg['patients'][pid])
    return df, fold_of, m[col].values.astype(float), D.seizures(df), D.segment_bounds(df)


def groupe_principal(sz):
    """Groupe de crises le plus fréquent du patient (connu à l'avance dans le produit)."""
    return sz['groupe'].value_counts().idxmax() if len(sz) else None


# ---------------------------------------------------------------- seuils choisis sans le bloc évalué
def _logit(x):
    x = np.clip(np.asarray(x, float), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def seuils_par_bloc(counts, heures, groupe_de, fa_cible, seuils=None):
    """Pour chaque (patient, bloc de test), seuil choisi sur les AUTRES blocs puis comptage sur ce bloc.

    counts : comptages d'une seule configuration (post-traitement, réfractaire, règle, périmètre) ;
    groupe_de : {patient: groupe principal}. Stratégies :
      global  : tous les patients, sauf le bloc évalué ;
      groupe  : patients du même groupe principal, sauf le bloc évalué ;
      patient : les 4 autres blocs du patient ;
      mixte   : moyenne (en logit) des seuils global et patient, ramenée au seuil de la grille le plus proche ;
      plafond : seuil global, relevé au seuil patient si celui-ci est plus haut (on ne baisse jamais le seuil global :
                seuls les patients qui ont trop de fausses alarmes sont corrigés).
    """
    seuils = np.sort(counts['seuil'].unique()) if seuils is None else np.asarray(seuils)
    pats = sorted(counts['patient_id'].unique()); blocs = sorted(counts['bloc'].unique())
    ip = {p: i for i, p in enumerate(pats)}; ib = {b: i for i, b in enumerate(blocs)}; it = {t: i for i, t in enumerate(seuils)}
    P, B, T = len(pats), len(blocs), len(seuils)
    fa = np.zeros((P, B, T)); det = np.zeros((P, B, T)); ncr = np.zeros((P, B))
    lat = {}
    for r in counts.itertuples(index=False):
        i, j, k = ip[r.patient_id], ib[r.bloc], it[r.seuil]
        fa[i, j, k] = r.n_fa; det[i, j, k] = r.n_detectees; ncr[i, j] = r.n_crises; lat[(i, j, k)] = r.latences
    hrs = np.zeros((P, B))
    for r in heures.itertuples(index=False):
        if r.patient_id in ip and r.bloc in ib:
            hrs[ip[r.patient_id], ib[r.bloc]] = r.heures
    grp = np.array([groupe_de.get(p) for p in pats], dtype=object)
    tot_fa, tot_h = fa.sum((0, 1)), hrs.sum()

    def choisir(fa_t, h):
        tab = pd.DataFrame({'seuil': seuils, 'n_fa': fa_t, 'heures': h})
        return seuil_pour_fa(tab, fa_cible)

    grp_fa = {g: fa[grp == g].sum((0, 1)) for g in set(grp)}
    grp_h = {g: hrs[grp == g].sum() for g in set(grp)}
    rows = []
    for i, p in enumerate(pats):
        for j, b in enumerate(blocs):
            if hrs[i, j] == 0:
                continue
            t = {'global': choisir(tot_fa - fa[i, j], tot_h - hrs[i, j]),
                 'groupe': choisir(grp_fa[grp[i]] - fa[i, j], grp_h[grp[i]] - hrs[i, j]),
                 'patient': choisir(fa[i].sum(0) - fa[i, j], hrs[i].sum() - hrs[i, j])}
            m = (_logit(t['global']) + _logit(t['patient'])) / 2
            t['mixte'] = float(seuils[np.abs(_logit(seuils) - m).argmin()])
            t['plafond'] = max(t['global'], t['patient'])
            for strat, thr in t.items():
                k = it[thr]
                rows.append({'strategie': strat, 'patient_id': p, 'bloc': b, 'seuil': thr, 'n_crises': ncr[i, j],
                             'n_detectees': det[i, j, k], 'n_fa': fa[i, j, k], 'heures': hrs[i, j],
                             'latences': list(lat.get((i, j, k), []))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- comparaisons
def sous(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).values
    return df[m]


def sens_a_fa(curve, fa):
    """Sensibilité interpolée à `fa` FA / 24 h ; NaN si la configuration n'atteint pas ce niveau de fausses alarmes."""
    if not len(curve) or fa < curve['fa_par_24h'].min():
        return np.nan
    return S.sens_at_fa(curve, fa)


def comparer(pp, cu, a, b, fa, n_boot=1000, seed=0):
    """Différence appariée de sensibilité (a - b) à ≈ `fa` FA / 24 h ; a, b : dicts de clés (post_traitement, ...)."""
    ta, sa = table_a_fa(sous(pp, **a), sous(cu, **a), fa)
    tb, sb = table_a_fa(sous(pp, **b), sous(cu, **b), fa)
    est, (lo, hi) = S.bootstrap_diff(ta, tb, n_boot=n_boot, seed=seed)
    return {'diff': est, 'ic_bas': lo, 'ic_haut': hi, 'seuil_a': sa, 'seuil_b': sb,
            'fa_a': ta['n_fa'].sum() / ta['heures'].sum() * 24, 'fa_b': tb['n_fa'].sum() / tb['heures'].sum() * 24}
