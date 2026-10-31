r"""Optionnel (non utilisé par le modèle v1) : écarts au contexte (30 min, 2 h), 10 retenus, ajoutés aux parquets de 05_patients.
Lancer après l'étape 5 : python optionnel/features_contexte.py

# Features d'écart au contexte (30 min et 2 h) : calcul, sélection des 10 meilleures, ajout aux parquets

**Contexte** : la première version (`preprocessing_contexte_features_v1_par_run.ipynb`) remettait le contexte à zéro
à chaque run ; or beaucoup de runs ne durent qu'une heure, la fenêtre longue était donc souvent incomplète.

**Pourquoi on ne repart pas du CSV source** : le CSV (`feat-grid-all-intersect_32patients.csv`) contient exactement
la même information temporelle que les parquets (chaque run démarre à `2000-01-01 00:00`, `elapsed_s` repart de 0,
aucune heure réelle) ; les parquets en sont une copie ligne à ligne. Le recalcul depuis le CSV ne changerait rien.

**Ce qui résout le problème : recoller les fichiers d'une heure.** En comparant les signaux de part et d'autre de
chaque frontière de run (section 1) : quand un run dure 56 à 62 min et que le suivant porte le numéro suivant, le
nouveau run reprend presque exactement là où le précédent s'arrêtait (orientation de la tête quasi identique, FC qui
ne saute que de ≈ 2 bpm) : ce sont des enregistrements continus découpés en fichiers d'une heure. On les recolle en
**segments continus** et le contexte est calculé sur ces segments. Les autres frontières (vraies interruptions,
sauts d'orientation 5 à 15 fois plus grands) remettent le contexte à zéro.

**Étapes**
1. Contrôle de la continuité des runs et construction des segments continus.
2. Calcul de **24 écarts au contexte candidats** (12 × 2 fenêtres : 30 min = 1800 s, 2 h = 7200 s), fenêtres causales
   (passé uniquement), sur les segments continus.
3. Lien de chaque candidat avec le label, **patient par patient** puis médiane sur les patients (pour qu'un patient
   comme sub-073 ne domine pas) : corrélation point-bisériale (corrélation de Pearson avec le label 0/1) et AUC
   univariée. Classement sur l'AUC (plus robuste qu'une corrélation avec un label présent dans 0,2 % des lignes),
   puis retrait des candidats trop redondants (|corrélation de Spearman| > 0,9 avec un candidat déjà retenu).
4. **Ajout des 10 retenus** aux parquets de `datas/data_with_seizure_id/` (réécriture atomique, colonnes `dev_*`
   remplacées si elles existent déjà). Les 24 candidats sont aussi sauvegardés dans
   `datas/contexte_features/candidats_30min_2h/`.

⚠ Les notebooks de modélisation qui prennent « toutes les colonnes sauf les métadonnées » incluront désormais ces
10 colonnes `dev_*` : pour reproduire un ancien résultat sans elles, les exclure explicitement.
"""

import os
import re
import glob
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sizeit import config as C  # noqa: E402

DATA_DIR = C.DATA_DIR                          # parquets par patient (étape 5)
CAND_DIR = os.path.join(C.WORK_DIR, 'optionnel_contexte', 'candidats_30min_2h')
os.makedirs(CAND_DIR, exist_ok=True)

HR, ACT, RMSSD, LFHF, SDNN = 'hrv_mean_hr', 'acc_EEG_SD_ACC_norme_1_std', 'hrv_rmssd', 'hrv_lf_hf_ratio', 'hrv_sdnn'
ORIENT = ['acc_EEG_SD_ACC_x_mean', 'acc_EEG_SD_ACC_y_mean', 'acc_EEG_SD_ACC_z_mean']
WINDOWS = {'30min': (1800, 300), '2h': (7200, 900)}   # nom : (taille en s, nb minimal de secondes disponibles)
HOUR_FILE_MIN_S, HOUR_FILE_MAX_S = 56 * 60, 62 * 60   # run « fichier d'une heure » recollé au suivant
N_SELECT = 10
REDUNDANCY_MAX = 0.9
files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))


def run_number(run_id):
    return int(re.search(r'run-(\d+)', run_id).group(1))

# ## 1. Continuité des runs et segments continus

bounds = []
for path in files:
    pid = os.path.splitext(os.path.basename(path))[0]
    d = pd.read_parquet(path, columns=['run_id', 'elapsed_s', HR] + ORIENT)
    d['rn'] = d['run_id'].map(run_number)
    d = d.sort_values(['rn', 'elapsed_s'], kind='stable')
    runs = [(rn, g) for rn, g in d.groupby('rn', sort=True)]
    for (ra, a), (rb, b) in zip(runs[:-1], runs[1:]):
        ha, hb = a[HR].dropna(), b[HR].dropna()
        bounds.append({'patient_id': pid, 'run_a': ra, 'run_b': rb, 'duree_a_min': (a['elapsed_s'].max() + 1) / 60,
                       'saut_orientation': float(np.linalg.norm(b[ORIENT].iloc[:10].mean().values - a[ORIENT].iloc[-10:].mean().values)),
                       'saut_fc_bpm': abs(hb.iloc[:30].mean() - ha.iloc[-30:].mean()) if len(ha) and len(hb) else np.nan})
bounds = pd.DataFrame(bounds)
bounds['recolle'] = ((bounds['run_b'] == bounds['run_a'] + 1) & (bounds['duree_a_min'] * 60 >= HOUR_FILE_MIN_S)
                     & (bounds['duree_a_min'] * 60 <= HOUR_FILE_MAX_S))
bounds.to_csv(os.path.join(CAND_DIR, '_frontieres_runs.csv'), index=False)
print(bounds.groupby('recolle').agg(n=('saut_orientation', 'size'), saut_orientation_median=('saut_orientation', 'median'),
                                      saut_fc_median=('saut_fc_bpm', 'median')).round(3))
print(f"{bounds['recolle'].sum()} frontières recollées sur {len(bounds)} "
      f"({bounds.groupby('patient_id')['recolle'].any().sum()} patients concernés)")

# ## 2. Calcul des 24 écarts candidats

def candidates_for_segment(s):
    out = {}
    hr, act, rmssd, lfhf, sdnn = s[HR], s[ACT], s[RMSSD], s[LFHF], s[SDNN]
    hr1, act1 = hr.rolling(60, min_periods=10).mean(), act.rolling(60, min_periods=10).mean()
    for name, (w, mp) in WINDOWS.items():
        r = lambda x: x.rolling(w, min_periods=mp)
        hr_m, hr_sd = r(hr).mean(), r(hr).std()
        act_m, act_sd = r(act).mean(), r(act).std()
        out[f'dev_fc_{name}'] = hr - hr_m
        out[f'dev_fc_z_{name}'] = (hr - hr_m) / hr_sd.replace(0, np.nan)
        out[f'dev_fc_repos_{name}'] = hr - r(hr).quantile(0.05)
        out[f'dev_fc_1min_{name}'] = hr1 - hr_m
        out[f'dev_agit_ratio_{name}'] = act / act_m.replace(0, np.nan)
        out[f'dev_agit_z_{name}'] = (act - act_m) / act_sd.replace(0, np.nan)
        out[f'dev_agit_1min_{name}'] = act1 / act_m.replace(0, np.nan)
        out[f'dev_agit_p95_{name}'] = act - r(act).quantile(0.95)
        out[f'dev_rmssd_{name}'] = rmssd - r(rmssd).median()
        out[f'dev_rmssd_z_{name}'] = (rmssd - r(rmssd).mean()) / r(rmssd).std().replace(0, np.nan)
        out[f'dev_lfhf_{name}'] = lfhf - r(lfhf).median()
        out[f'dev_sdnn_{name}'] = sdnn - r(sdnn).median()
    return pd.DataFrame({k: v.values.astype(np.float32) for k, v in out.items()}, index=s.index)


def auc_univariate(x, y):
    m = ~np.isnan(x)
    x, y = x[m], y[m]
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    rk = rankdata(x)
    return (rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


recolle = set(map(tuple, bounds.loc[bounds['recolle'], ['patient_id', 'run_b']].values))
stats_rows, samples = [], []
t0 = time.time()
for path in files:
    pid = os.path.splitext(os.path.basename(path))[0]
    d = pd.read_parquet(path, columns=['run_id', 'elapsed_s', 'label', HR, ACT, RMSSD, LFHF, SDNN])
    d['rn'] = d['run_id'].map(run_number)
    order = d.sort_values(['rn', 'elapsed_s'], kind='stable').index
    ds = d.loc[order]
    new_seg = np.r_[True, ds['rn'].values[1:] != ds['rn'].values[:-1]]
    new_seg &= ~np.array([(pid, rn) in recolle for rn in ds['rn'].values])   # frontière recollée : même segment
    ds['segment'] = np.cumsum(new_seg)
    cand = pd.concat([candidates_for_segment(g) for _, g in ds.groupby('segment', sort=False)])
    cand = cand.reindex(d.index)                                   # retour à l'ordre des lignes du fichier
    cand.to_parquet(os.path.join(CAND_DIR, f'{pid}.parquet'), index=False)
    y = d['label'].fillna(0).values.astype(int)
    if y.sum():
        for c in cand.columns:
            x = cand[c].values.astype(float)
            ok = ~np.isnan(x)
            stats_rows.append({'patient_id': pid, 'feature': c, 'auc': auc_univariate(x, y),
                               'corr_pointbiseriale': np.corrcoef(x[ok], y[ok])[0, 1] if y[ok].std() > 0 else np.nan,
                               'manquant': 1 - ok.mean()})
    rng = np.random.default_rng(0)
    samples.append(cand.iloc[rng.choice(len(cand), min(len(cand), 20000), replace=False)])
    print(f"{pid} : {ds['segment'].max()} segments continus pour {d['run_id'].nunique()} runs ({time.time() - t0:.0f}s)",
          flush=True)
stats = pd.DataFrame(stats_rows)

# ## 3. Sélection des 10 écarts les plus liés au label

agg = stats.groupby('feature').agg(auc_mediane=('auc', 'median'), corr_mediane=('corr_pointbiseriale', 'median'),
                                   manquant_moy=('manquant', 'mean'),
                                   auc_q25=('auc', lambda s: s.quantile(0.25)), auc_q75=('auc', lambda s: s.quantile(0.75)))
agg['pouvoir_discriminant'] = (agg['auc_mediane'] - 0.5).abs() * 2          # 0 = aucun lien, 1 = séparation parfaite
agg['sens'] = np.where(agg['auc_mediane'] >= 0.5, 'plus élevé en crise', 'plus bas en crise')
# même sens chez combien de patients
same_dir = stats.assign(up=stats['auc'] > 0.5).groupby('feature')['up'].mean()
agg['part_patients_meme_sens'] = np.where(agg['auc_mediane'] >= 0.5, same_dir, 1 - same_dir).round(2)
agg = agg.sort_values('pouvoir_discriminant', ascending=False)

sample = pd.concat(samples)
selected, rejected = [], {}
for f in agg.index:
    redundant = [g for g in selected if abs(spearmanr(sample[f], sample[g], nan_policy='omit').statistic) > REDUNDANCY_MAX]
    if redundant:
        rejected[f] = redundant[0]
        continue
    selected.append(f)
    if len(selected) == N_SELECT:
        break
agg['retenue'] = agg.index.isin(selected)
agg['redondante_avec'] = agg.index.map(rejected)
agg.to_csv(os.path.join(CAND_DIR, '_classement_candidats.csv'))
print('Écarts retenus :', selected)
print(agg.round(3))

# ## 4. Ajout des 10 écarts retenus aux parquets `data_with_seizure_id`

t0 = time.time()
for path in files:
    pid = os.path.splitext(os.path.basename(path))[0]
    df = pd.read_parquet(path)
    cand = pd.read_parquet(os.path.join(CAND_DIR, f'{pid}.parquet'), columns=selected)
    assert len(cand) == len(df), pid
    df = df.drop(columns=[c for c in df.columns if c.startswith('dev_')])      # idempotent
    df = pd.concat([df, cand.set_index(df.index)], axis=1)
    tmp = path + '.tmp'
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)                                                      # remplacement atomique
    print(f'{pid} : {len(selected)} colonnes ajoutées ({time.time() - t0:.0f}s)', flush=True)

check = pd.read_parquet(files[0])
print([c for c in check.columns if c.startswith('dev_')])
