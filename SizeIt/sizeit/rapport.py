"""Rapport type d'un modèle : un PDF standard + tableaux CSV, produit à partir des prédictions par seconde.

Entrée : DataFrame de prédictions hors échantillon avec au moins
    patient_id, run_id, elapsed_s, proba
(une ligne par seconde ; les labels, crises, segments et l'ordre chronologique sont relus avec donnees.load_patient,
ce qui garantit la même vérité terrain et la même règle de comptage pour tous les modèles).

Usage :
    from lib import rapport
    res = rapport.generer(preds, out_dir, meta={'nom': 'baseline hrv_norme_custom', ...},
                          postprocess={'method': 'kieme_max', 'n': 20, 'k': 5}, seuil=0.5, in_scope=None,
                          importance=None, reference=None)
`reference` : résultat d'un autre appel à `generer` (pour la comparaison à fausses alarmes égales).
"""
import os
import json
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from sklearn.calibration import calibration_curve

from . import config as C
from . import donnees as D
from . import scoring as S

THRESHOLDS = np.round(np.r_[np.arange(0.05, 0.951, 0.05), 0.975, 0.99], 3)
FA_LEVELS = [1, 2, 3, 5]
INK, INK2, BLUE, ORANGE, RED, GREY = '#0b0b0b', '#52514e', '#2a78d6', '#eb6834', '#e34948', '#a8a6a0'
CM = LinearSegmentedColormap.from_list('cm', ['#f4f8fd', '#86b6ef', '#1c5cab'])


# ---------------------------------------------------------------- calcul
def _align(pid, p):
    """Prédictions du patient remises dans l'ordre chronologique de load_patient (positions, segments, crises)."""
    df = D.load_patient(pid, columns=[])
    key = df[['run_id', 'elapsed_s']].astype({'run_id': str})
    p = p[['run_id', 'elapsed_s', 'proba']].astype({'run_id': str})
    m = key.merge(p, on=['run_id', 'elapsed_s'], how='left', validate='one_to_one')
    if m['proba'].isna().any():
        raise ValueError(f'{pid} : {int(m["proba"].isna().sum())} secondes sans prédiction')
    df['proba'] = m['proba'].values
    return df


def _patient_eval(df, sz, segb, pp, seuils, in_scope):
    score = S.postprocess(df['proba'].values, df['segment'].values, **pp)
    out = {}
    for rule in C.SCORING:
        for thr in seuils:
            ev = S.alarms(score, df['segment'].values, thr)
            s, a = S.score_patient(ev, sz, segb, rule=rule, in_scope=in_scope)
            out[(rule, thr)] = (s, a)
    return score, out


def seuil_de(seuil, pid):
    # seuil global (float) ou seuil propre à chaque patient (dict patient -> seuil, ex. fixé par validation interne)
    return float(seuil[pid]) if isinstance(seuil, dict) else float(seuil)


def seuil_txt(seuil):
    if isinstance(seuil, dict):
        v = np.array(list(seuil.values()), dtype=float)
        return f'propre à chaque patient (médiane {np.median(v):.3f}, de {v.min():.3f} à {v.max():.3f})'
    return f'{seuil}'


def evaluer(preds, postprocess=None, seuil=0.5, in_scope=None, verbose=True):
    pp = postprocess or {'method': 'moyenne', 'n': 10}
    rows, sz_rows, al_rows, sec = [], [], [], {}
    for pid, p in preds.groupby('patient_id'):
        thr_p = seuil_de(seuil, pid)
        seuils = sorted(set(THRESHOLDS.tolist() + [thr_p]))
        df = _align(pid, p)
        sz = D.seizures(df)
        segb = D.segment_bounds(df)
        score, out = _patient_eval(df, sz, segb, pp, seuils, in_scope)
        hours = len(df) / 3600
        for (rule, thr), (s, a) in out.items():
            rows.append({'patient_id': pid, 'regle': rule, 'seuil': thr, 'n_crises': len(s),
                         'n_detectees': int(s['detectee'].sum()) if len(s) else 0,
                         'n_fa': int((a['statut'] == 'fausse alarme').sum()), 'heures': hours,
                         'latences': s.loc[s['detectee'] == True, 'latence_s'].tolist() if len(s) else [],
                         'fa_duree_s': int(a.loc[a['statut'] == 'fausse alarme', 'duree_s'].sum()),
                         'n_fa_post_ictales': int(a['post_ictale'].sum()) if len(a) else 0,
                         'operation': thr == thr_p})
            if thr == thr_p:
                sz_rows.append(s.assign(patient_id=pid, regle=rule))
                al_rows.append(a.assign(patient_id=pid, regle=rule))
        y = df['label'].values
        if in_scope is not None:
            keep = (y == 0) | df['groupe'].isin(in_scope).values
        else:
            keep = np.ones(len(df), bool)
        sec[pid] = {'y': y[keep], 'p': df['proba'].values[keep], 'score': score[keep],
                    'df': df[['rn', 'elapsed_s', 'segment', 'label', 'proba']].assign(score=score)}
        if verbose:
            print(f'  {pid} évalué', flush=True)
    per = pd.DataFrame(rows)
    return {'par_patient_seuil': per, 'crises': pd.concat(sz_rows, ignore_index=True),
            'alarmes': pd.concat(al_rows, ignore_index=True), 'secondes': sec, 'seuil': seuil,
            'postprocess': pp, 'in_scope': in_scope}


def _curve(per, rule):
    g = per[(per['regle'] == rule) & per['seuil'].isin(THRESHOLDS)]
    return pd.DataFrame([{'seuil': thr, **S.summarize(d)} for thr, d in g.groupby('seuil')])


# ---------------------------------------------------------------- pages
def _text_page(pdf, title, blocks, fontsize=9.5, width=120):
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.text(0.04, 0.95, title, fontsize=15, fontweight='bold', color=INK, va='top')
    y = 0.885
    for style, txt in blocks:
        if style == 'h':
            y -= 0.01; fig.text(0.04, y, txt, fontsize=fontsize + 1.5, fontweight='bold', color='#1c5cab', va='top'); y -= 0.04
            continue
        prefix, indent = ('•  ', 0.055) if style == 'b' else ('', 0.04)
        for j, line in enumerate(textwrap.wrap(str(txt), width=width - (4 if style == 'b' else 0))):
            fig.text(indent, y, (prefix if j == 0 else '   ') + line, fontsize=fontsize, color=INK, va='top'); y -= 0.029
        y -= 0.007
    pdf.savefig(fig); plt.close(fig)


def _table(ax, cells, hdr, widths=None, fs=8, scale=1.3, colors=None):
    t = ax.table(cellText=cells, colLabels=hdr, loc='upper left', cellLoc='center', colWidths=widths)
    t.auto_set_font_size(False); t.set_fontsize(fs); t.scale(1, scale)
    for (r_, c_), cell in t.get_celld().items():
        cell.set_edgecolor('#d9d7d2')
        if r_ == 0:
            cell.set_text_props(fontweight='bold'); cell.set_facecolor('#f2f6fc')
        elif colors is not None:
            cell.set_facecolor(colors[r_ - 1])


def _window(ax, d, s, e, title, shade=None, seuil=0.5):
    seg = d['segment'].values[s]
    in_seg = np.flatnonzero(d['segment'].values == seg)
    a, b = max(in_seg[0], s - 180), min(in_seg[-1] + 1, e + 180)
    t = np.arange(a, b) - s
    ax.fill_between(t, 0, 1, where=d['label'].values[a:b] == 1, color=RED, alpha=0.15, step='post', lw=0)
    if shade is not None:
        ax.axvspan(shade[0] - s, shade[1] - s, color=ORANGE, alpha=0.2, lw=0)
    ax.plot(t, d['proba'].values[a:b], color='#9ec5f4', lw=0.7)
    ax.plot(t, d['score'].values[a:b], color=BLUE, lw=1.3)
    ax.axhline(seuil, color=INK2, ls='--', lw=0.7)
    ax.set_ylim(0, 1.02); ax.set_xlim(t[0], t[-1]); ax.tick_params(labelsize=6.5)
    ax.set_title(title, fontsize=7, loc='left')


def generer(preds, out_dir, meta, postprocess=None, seuil=0.5, in_scope=None, importance=None, reference=None,
            n_boot=2000, verbose=False):
    os.makedirs(out_dir, exist_ok=True)
    R = evaluer(preds, postprocess, seuil, in_scope, verbose)
    per, crises, alarmes, sec = R['par_patient_seuil'], R['crises'], R['alarmes'], R['secondes']
    curves = {rule: _curve(per, rule) for rule in C.SCORING}
    op = {rule: per[(per['regle'] == rule) & per['operation']] for rule in C.SCORING}
    summ = {rule: S.summarize(op[rule]) for rule in C.SCORING}
    ci = {rule: S.bootstrap_ci(op[rule], n_boot=n_boot) for rule in C.SCORING}
    matched = {rule: {fa: S.sens_at_fa(curves[rule], fa) for fa in FA_LEVELS} for rule in C.SCORING}
    y_all = np.concatenate([v['y'] for v in sec.values()]); p_all = np.concatenate([v['p'] for v in sec.values()])
    sec_pat = pd.DataFrame([{'patient_id': pid, 'roc_auc': roc_auc_score(v['y'], v['p']) if 0 < v['y'].sum() < len(v['y']) else np.nan,
                             'pr_auc': average_precision_score(v['y'], v['p']) if v['y'].sum() else np.nan,
                             'prevalence': v['y'].mean()} for pid, v in sec.items()])
    sec_glob = {'roc_auc_global': roc_auc_score(y_all, p_all), 'pr_auc_global': average_precision_score(y_all, p_all),
                'roc_auc_mediane_patients': sec_pat['roc_auc'].median(), 'pr_auc_mediane_patients': sec_pat['pr_auc'].median()}

    # ---- tableaux
    crises_std = crises[crises['regle'] == 'standard']
    crises_std = crises_std.assign(detectee=crises_std['detectee'].astype(float))
    by = lambda col: (crises_std.groupby(col)['detectee'].agg(['sum', 'size'])
                      .assign(sensibilite=lambda d: d['sum'] / d['size'],
                              ic_bas=lambda d: [S.wilson(k, n)[0] for k, n in zip(d['sum'], d['size'])],
                              ic_haut=lambda d: [S.wilson(k, n)[1] for k, n in zip(d['sum'], d['size'])],
                              latence_mediane_s=crises_std.groupby(col)['latence_s'].median()))
    t_groupe, t_type = by('groupe'), by('type')
    t_pat = op['standard'][['patient_id', 'n_crises', 'n_detectees', 'n_fa', 'heures', 'fa_duree_s', 'n_fa_post_ictales']].copy()
    t_pat['sensibilite'] = t_pat['n_detectees'] / t_pat['n_crises'].replace(0, np.nan)
    t_pat['fa_par_24h'] = t_pat['n_fa'] / t_pat['heures'] * 24
    t_pat = t_pat.merge(sec_pat, on='patient_id').merge(
        op['strict'][['patient_id', 'n_detectees']].rename(columns={'n_detectees': 'n_detectees_strict'}), on='patient_id')
    resume = {rule: {**summ[rule], **{f'ic95_{k}': v for k, v in ci[rule].items()},
                     **{f'sens@{fa}FA': matched[rule][fa] for fa in FA_LEVELS}} for rule in C.SCORING}
    with open(os.path.join(out_dir, 'resume.json'), 'w') as fh:
        json.dump({'meta': meta, 'seuil': seuil, 'postprocess': R['postprocess'], 'in_scope': in_scope,
                   'evenements': resume, 'secondes': sec_glob}, fh, indent=1, ensure_ascii=False, default=float)
    t_groupe.to_csv(os.path.join(out_dir, 'detection_par_groupe.csv'))
    t_type.to_csv(os.path.join(out_dir, 'detection_par_type.csv'))
    t_pat.to_csv(os.path.join(out_dir, 'detection_par_patient.csv'), index=False)
    for rule in C.SCORING:
        curves[rule].to_csv(os.path.join(out_dir, f'courbe_seuils_{rule}.csv'), index=False)
    crises.to_csv(os.path.join(out_dir, 'crises_detail.csv'), index=False)
    alarmes.to_csv(os.path.join(out_dir, 'alarmes_detail.csv'), index=False)

    # ---- PDF
    pdf_path = os.path.join(out_dir, 'rapport.pdf')
    pp = R['postprocess']
    pp_txt = (f"moyenne des {pp.get('n', 10)} dernières secondes" if pp['method'] == 'moyenne' else
              f"{pp.get('k')}e plus grande probabilité des {pp.get('n')} dernières secondes" if pp['method'] == 'kieme_max'
              else 'probabilité brute')
    with PdfPages(pdf_path) as pdf:
        # 1. identité + résultats principaux
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(0.04, 0.95, f"Rapport : {meta.get('nom', 'modèle')}", fontsize=15, fontweight='bold', va='top')
        ident = [f'{k} : {v}' for k, v in meta.items() if k != 'nom'] + [
            f'post-traitement : {pp_txt}', f'seuil : {seuil_txt(seuil)}',
            f"crises ciblées : {', '.join(in_scope) if in_scope else 'toutes'}",
            f"patients : {summ['standard']['n_patients']}, crises ciblées : {summ['standard']['n_crises']}, "
            f"heures : {op['standard']['heures'].sum():,.0f}".replace(',', ' ')]
        y = 0.89
        for line in ident:
            for j, l in enumerate(textwrap.wrap(line, 150)):
                fig.text(0.04, y, ('' if j == 0 else '   ') + l, fontsize=8.5, color=INK2, va='top'); y -= 0.025
        ax = fig.add_axes([0.04, 0.08, 0.92, y - 0.12]); ax.axis('off')
        cells = []
        for rule in C.SCORING:
            s, c = summ[rule], ci[rule]
            cells.append([rule, f"{s['n_detectees']}/{s['n_crises']}",
                          f"{s['sensibilite']:.0%} [{c['sensibilite'][0]:.0%} ; {c['sensibilite'][1]:.0%}]",
                          f"{s['fa_par_24h']:.2f} [{c['fa_par_24h'][0]:.2f} ; {c['fa_par_24h'][1]:.2f}]",
                          f"{s['sensibilite_mediane_patients']:.0%}", f"{s['part_patients_au_moins_une']:.0%}",
                          f"{s['latence_mediane_s']:.0f} / {s['latence_p90_s']:.0f} s",
                          ' '.join(f"{matched[rule][fa]:.0%}" for fa in FA_LEVELS)])
        _table(ax, cells, ['Règle', 'Crises dét.', 'Sensibilité [IC 95 %]', 'FA / 24 h [IC 95 %]', 'Sens. méd. pat.',
                           'Pat. ≥ 1 dét.', 'Latence méd./P90', 'Sens. @1/2/3/5 FA'],
               [0.08, 0.08, 0.17, 0.17, 0.11, 0.1, 0.12, 0.17], fs=8.5, scale=2)
        fig.text(0.04, 0.05, "Standard : alarme chevauchant [début - 30 s, fin + 60 s]. Strict : alarme ≥ 10 s chevauchant la crise. "
                 "IC 95 % par rééchantillonnage des patients. Sens. @x FA : seuil réglé pour x fausses alarmes / 24 h. "
                 f"Seconde par seconde : ROC-AUC {sec_glob['roc_auc_global']:.3f}, PR-AUC {sec_glob['pr_auc_global']:.3f} "
                 f"(médianes patients {sec_glob['roc_auc_mediane_patients']:.3f} / {sec_glob['pr_auc_mediane_patients']:.3f}).",
                 fontsize=7.5, color=INK2, wrap=True)
        pdf.savefig(fig); plt.close(fig)

        # 2. courbes sensibilité / FA
        fig, axes = plt.subplots(1, 2, figsize=(11.7, 8.3))
        for rule, col in (('standard', BLUE), ('strict', ORANGE)):
            c = curves[rule].sort_values('fa_par_24h')
            axes[0].plot(c['fa_par_24h'], c['sensibilite'] * 100, marker='o', ms=3, color=col, label=rule)
            o = summ[rule]; axes[0].scatter(o['fa_par_24h'], o['sensibilite'] * 100, s=80, facecolor='none', edgecolor=col, lw=1.5)
            if reference is not None:
                rc = reference['courbes'][rule].sort_values('fa_par_24h')
                axes[0].plot(rc['fa_par_24h'], rc['sensibilite'] * 100, color=col, ls=':', lw=1, label=f'{rule} (référence)')
        axes[0].set_xscale('log'); axes[0].set_xlim(0.2, 50); axes[0].set_ylim(0, 100)
        axes[0].set_xticks([0.5, 1, 2, 3, 5, 10, 20], ['0,5', '1', '2', '3', '5', '10', '20'])
        axes[0].set_xlabel('Fausses alarmes / 24 h'); axes[0].set_ylabel('Crises détectées (%)')
        axes[0].grid(color='#e6e5e1', lw=0.6); axes[0].legend(fontsize=8)
        axes[0].set_title('Compromis détection / fausses alarmes (cercle = seuil retenu)', loc='left', fontsize=10, fontweight='bold')
        tg =t_groupe.reindex([x for x in C.SEIZURE_GROUPS if x in t_groupe.index])
        axes[1].barh(range(len(tg)), tg['sensibilite'] * 100, color=BLUE, height=0.6,
                     xerr=[(tg['sensibilite'] - tg['ic_bas']) * 100, (tg['ic_haut'] - tg['sensibilite']) * 100],
                     error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
        for yi, (k, n) in enumerate(zip(tg['sum'], tg['size'])):
            axes[1].text(102, yi, f'{int(k)}/{int(n)}', va='center', fontsize=8)
        axes[1].set_yticks(range(len(tg)), [textwrap.fill(f"{x} ({C.CATEGORIES[x]})", 22) for x in tg.index], fontsize=8)
        axes[1].invert_yaxis(); axes[1].set_xlim(0, 115); axes[1].set_xlabel('Crises détectées (%), règle standard, IC 95 % Wilson')
        axes[1].set_title('Détection par groupe de crises', loc='left', fontsize=10, fontweight='bold')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 3. par type et par patient
        fig, axes = plt.subplots(1, 2, figsize=(11.7, 8.3), gridspec_kw={'width_ratios': [1, 1.4]})
        axes[0].axis('off')
        tt = t_type.sort_values('size', ascending=False)
        _table(axes[0], [[t, f"{int(r['sum'])}/{int(r['size'])}", f"{r['sensibilite']:.0%}",
                          f"[{r['ic_bas']:.0%} ; {r['ic_haut']:.0%}]", '-' if pd.isna(r['latence_mediane_s']) else f"{r['latence_mediane_s']:.0f}"]
                         for t, r in tt.iterrows()], ['Type', 'Dét.', 'Sens.', 'IC 95 %', 'Lat. (s)'],
               [0.38, 0.14, 0.14, 0.22, 0.12], fs=7.5, scale=1.2)
        axes[0].set_title('Par type de crise (règle standard)', loc='left', fontsize=10, fontweight='bold')
        axes[1].axis('off')
        tp = t_pat.sort_values(['n_crises'], ascending=False)
        _table(axes[1], [[r['patient_id'], f"{int(r['n_detectees'])}/{int(r['n_crises'])}", f"{int(r['n_detectees_strict'])}",
                          f"{r['fa_par_24h']:.1f}", f"{int(r['n_fa_post_ictales'])}", f"{r['fa_duree_s'] / 60:.0f}",
                          f"{r['pr_auc']:.3f}", f"{r['roc_auc']:.3f}"] for _, r in tp.iterrows()],
               ['Patient', 'Dét. std', 'Dét. strict', 'FA/24h', 'FA post-ict.', 'Durée FA (min)', 'PR-AUC', 'ROC-AUC'],
               [0.13, 0.11, 0.11, 0.1, 0.12, 0.14, 0.12, 0.12], fs=6.8, scale=1.0)
        axes[1].set_title('Par patient (seuil retenu)', loc='left', fontsize=10, fontweight='bold')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 4. distributions par patient + seconde par seconde
        fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))
        sp = t_pat.dropna(subset=['sensibilite'])
        axes[0, 0].hist(sp['sensibilite'] * 100, bins=np.arange(0, 105, 10), color=BLUE, edgecolor='white')
        axes[0, 0].set_xlabel('Sensibilité du patient (%)'); axes[0, 0].set_ylabel('Patients')
        axes[0, 0].set_title('Distribution des sensibilités par patient', loc='left', fontsize=10, fontweight='bold')
        axes[0, 1].bar(range(len(tp)), tp['fa_par_24h'], color=ORANGE)
        axes[0, 1].set_xticks(range(len(tp)), tp['patient_id'], rotation=90, fontsize=6)
        axes[0, 1].set_ylabel('FA / 24 h'); axes[0, 1].set_title('Fausses alarmes par patient', loc='left', fontsize=10, fontweight='bold')
        pr, rc, _ = precision_recall_curve(y_all, p_all); fpr, tpr, _ = roc_curve(y_all, p_all)
        axes[1, 0].plot(rc, pr, color=BLUE, label=f"PR (AP {sec_glob['pr_auc_global']:.3f})")
        axes[1, 0].plot(fpr, tpr, color=GREY, label=f"ROC (AUC {sec_glob['roc_auc_global']:.3f})")
        axes[1, 0].set_xlabel('rappel / taux de faux positifs'); axes[1, 0].legend(fontsize=8)
        axes[1, 0].set_title('Seconde par seconde (toutes secondes poolées)', loc='left', fontsize=10, fontweight='bold')
        rng = np.random.default_rng(0)
        idx = np.r_[np.flatnonzero(y_all == 1), rng.choice(np.flatnonzero(y_all == 0), min(200000, int((y_all == 0).sum())), replace=False)]
        frac, mean_p = calibration_curve(y_all[idx], p_all[idx], n_bins=10, strategy='quantile')
        axes[1, 1].plot(mean_p, frac, marker='o', color=BLUE, label='échantillon (crises sur-représentées)')
        axes[1, 1].plot([0, 1], [0, 1], color=GREY, ls='--', lw=0.8)
        axes[1, 1].set_xlabel('probabilité prédite'); axes[1, 1].set_ylabel('fréquence observée')
        axes[1, 1].set_title('Calibration (diagnostic, sur sous-échantillon)', loc='left', fontsize=10, fontweight='bold')
        axes[1, 1].legend(fontsize=7)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 5. analyse des erreurs
        fa = alarmes[(alarmes['regle'] == 'standard') & (alarmes['statut'] == 'fausse alarme')]
        missed = crises_std[crises_std['detectee'] == False]
        fig, axes = plt.subplots(1, 3, figsize=(11.7, 8.3))
        axes[0].hist(np.clip(fa['duree_s'], 1, 3600), bins=np.logspace(0, np.log10(3600), 25), color=ORANGE, edgecolor='white')
        axes[0].set_xscale('log'); axes[0].set_xlabel('durée de la fausse alarme (s)')
        axes[0].set_title(f"Fausses alarmes : {len(fa)}\n{fa['post_ictale'].mean():.0%} post-ictales (< 1 h après une crise)",
                          loc='left', fontsize=10, fontweight='bold')
        if len(crises_std):
            det = crises_std.assign(classe=pd.cut(crises_std['duree_s'], [0, 30, 60, 120, 300, 1e9],
                                                  labels=['< 30 s', '30-60 s', '1-2 min', '2-5 min', '> 5 min']))
            rr = det.groupby('classe', observed=False)['detectee'].agg(['sum', 'size'])
            axes[1].bar(range(len(rr)), rr['sum'] / rr['size'].replace(0, np.nan) * 100, color=BLUE)
            for xi, (k, n) in enumerate(zip(rr['sum'], rr['size'])):
                axes[1].text(xi, 2, f'{int(k)}/{int(n)}', ha='center', fontsize=8, color='white')
            axes[1].set_xticks(range(len(rr)), rr.index.astype(str), fontsize=8)
            axes[1].set_ylabel('crises détectées (%)'); axes[1].set_ylim(0, 105)
            axes[1].set_title('Détection selon la durée de la crise', loc='left', fontsize=10, fontweight='bold')
        lat = crises_std['latence_s'].dropna()
        axes[2].hist(lat, bins=20, color=BLUE, edgecolor='white'); axes[2].axvline(0, color=INK, lw=0.8)
        axes[2].set_xlabel('latence (s, négatif = alarme avant le début annoté)')
        axes[2].set_title('Latence de détection', loc='left', fontsize=10, fontweight='bold')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        for title, items in (('Crises manquées (± 3 min)', missed.sort_values('duree_s', ascending=False).head(12)),
                             ('Fausses alarmes les plus longues (± 3 min)', fa.sort_values('duree_s', ascending=False).head(12))):
            if not len(items):
                continue
            fig, axes = plt.subplots(4, 3, figsize=(11.7, 8.3))
            for ax, (_, r) in zip(axes.flat, items.iterrows()):
                d = sec[r['patient_id']]['df']
                if 'seizure_id' in r and isinstance(r.get('seizure_id'), str):
                    _window(ax, d, int(r['start']), int(r['end']), f"{r['patient_id']} {r['seizure_id']} {r['type']} {int(r['duree_s'])} s", seuil=seuil_de(seuil, r['patient_id']))
                else:
                    _window(ax, d, int(r['start']), int(r['end']),
                            f"{r['patient_id']} | {int(r['duree_s'])} s | {'post-ictale' if r['post_ictale'] else ''}",
                            shade=(int(r['start']), int(r['end'])), seuil=seuil_de(seuil, r['patient_id']))
            for ax in axes.flat[len(items):]:
                ax.set_visible(False)
            fig.suptitle(title + ' : probabilité brute (bleu clair), score post-traité (bleu), crise (rouge)',
                         x=0.01, ha='left', fontweight='bold', fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.95)); pdf.savefig(fig); plt.close(fig)

        # 6. importance des features (optionnelle)
        if importance is not None and len(importance):
            imp = importance.sort_values('importance', ascending=False).head(30).iloc[::-1]
            fig, ax = plt.subplots(figsize=(11.7, 8.3))
            cols = {'HRV': RED, 'ACC norme': BLUE, 'ACC bandes custom': '#1baf7a', 'ACC bandes 1 Hz': '#eda100',
                    'ACC axes x/y/z': GREY, 'Écart au contexte': '#e87ba4'}
            ax.barh(imp['feature'].str.replace('acc_EEG_SD_ACC_', 'acc_'), imp['importance'],
                    color=[cols[C.feature_family(f)] for f in imp['feature']])
            ax.set_xlabel('importance'); ax.tick_params(axis='y', labelsize=7.5)
            ax.set_title('Les 30 features les plus importantes', loc='left', fontsize=11, fontweight='bold')
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 7. comparaison à une référence (optionnelle), à fausses alarmes égales, appariée par patient
        if reference is not None:
            lines = []
            for rule in C.SCORING:
                for fa_lvl in (2, 3, 5):
                    a, b = matched[rule][fa_lvl], S.sens_at_fa(reference['courbes'][rule], fa_lvl)
                    lines.append(('b', f"{rule}, {fa_lvl} FA / 24 h : {a:.0%} contre {b:.0%} pour la référence ({a - b:+.0%})"))
            _text_page(pdf, f"Comparaison à la référence : {reference['meta'].get('nom', '')}", lines)

    res = {'meta': meta, 'resume': resume, 'courbes': curves, 'par_patient': t_pat, 'par_groupe': t_groupe,
           'par_type': t_type, 'secondes': sec_glob, 'pdf': pdf_path, 'par_patient_seuil': per, 'crises': crises}
    return res
