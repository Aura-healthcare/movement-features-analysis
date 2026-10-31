r"""Optionnel (non utilisé par le modèle v1) : features CL_Fourier, ajoutées aux parquets de 05_patients.
Lancer après l'étape 5 : python optionnel/features_cl_fourier.py

# Features CL_Fourier : puissance du mouvement dans les bandes 2–7 Hz et 7–12,5 Hz

Calcul à partir de l'accélération brute (`datas/Datas_SizeIT/csv_acc_renormalised`, 25 Hz, axes EEG SD ACC X / Y / Z).

**Méthode** (pour chaque seconde *t* de chaque run) :
1. fenêtre de **10 s centrée** sur la seconde *t* (même convention que les features existantes, calculées sur
   10 s et datées au milieu de la fenêtre) ;
2. sur chaque axe : densité spectrale de puissance par la **méthode de Welch** (segments de 4 s = 100 échantillons,
   chevauchement 50 %, fenêtre de Hann, moyenne retirée par segment), unilatérale, en g²/Hz :
   $P[k] = \frac{2}{f_s \sum w^2} |X[k]|^2$ (sans le facteur 2 pour 0 Hz et 12,5 Hz) ;
3. **somme des trois axes** (l'énergie du vecteur = somme des énergies des axes : indépendant de l'orientation
   du capteur ; la gravité est retirée par le centrage) ;
4. intégration sur la bande : $P_{bande} = \sum_{f_k \in bande} P[k] \, \Delta f$ avec $\Delta f = 0,25$ Hz, en g².
   - `CL_Fourier_2_7` : 2 ≤ f < 7 Hz ;
   - `CL_Fourier_7_125` : 7 ≤ f ≤ 12,5 Hz (12,5 Hz = fréquence de Nyquist).

Les 5 premières et 5 dernières secondes de chaque run (fenêtre incomplète) valent NaN.

⚠ 12,5 Hz est la limite de ce qu'un signal à 25 Hz peut contenir : la bande 7–12,5 Hz dépend du filtre
anti-repliement du capteur (inconnu).

**Sorties** : `datas/CL_fourier_features/<patient>.parquet` (run_id, elapsed_s, 2 features), puis ajout des 2 colonnes
aux parquets de `datas/data_with_seizure_id/` (réécriture atomique, colonnes existantes vérifiées inchangées).
"""

import os
import glob
import time
import hashlib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from concurrent.futures import ProcessPoolExecutor

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sizeit import config as C  # noqa: E402

DATA_DIR = C.DATA_DIR                          # parquets par patient (étape 5)
RAW_DIR = C.CSV_ACC_RENORM_DIR                 # accélération renormalisée (étape 2)
OUT_DIR = os.path.join(C.WORK_DIR, 'optionnel_CL_fourier')
os.makedirs(OUT_DIR, exist_ok=True)

FS = 25
WIN = 10 * FS              # fenêtre de 10 s
SEG, STEP = 4 * FS, 2 * FS  # Welch : segments de 4 s, chevauchement 50 %
AXES = ['EEG SD ACC X', 'EEG SD ACC Y', 'EEG SD ACC Z']
BANDES = {'CL_Fourier_2_7': (2.0, 7.0, False), 'CL_Fourier_7_125': (7.0, 12.5, True)}   # (bas, haut, haut inclus)
N_WORKERS = 10

HANN = np.hanning(SEG + 1)[:-1]                        # fenêtre de Hann périodique (comme scipy)
SCALE = 2.0 / (FS * (HANN ** 2).sum())
FREQS = np.fft.rfftfreq(SEG, 1 / FS)                   # 0 ; 0,25 ; ... ; 12,5 Hz
DF = FREQS[1] - FREQS[0]
MASKS = {k: (FREQS >= lo) & ((FREQS <= hi) if inc else (FREQS < hi)) for k, (lo, hi, inc) in BANDES.items()}


def band_powers(sig):
    # sig : (n_fenetres, WIN) d'un axe -> puissance de chaque bande (n_fenetres,)
    offs = range(0, WIN - SEG + 1, STEP)                                  # 4 segments par fenêtre de 10 s
    psd = 0
    for o in offs:
        s = sig[:, o:o + SEG]
        s = (s - s.mean(axis=1, keepdims=True)) * HANN
        p = np.abs(np.fft.rfft(s, axis=1)) ** 2 * SCALE
        p[:, 0] /= 2; p[:, -1] /= 2                                        # 0 Hz et Nyquist : pas de facteur 2
        psd = psd + p
    psd /= len(offs)
    return {k: psd[:, m].sum(axis=1) * DF for k, m in MASKS.items()}


def cl_fourier_run(run_id, elapsed):
    # valeurs pour chaque seconde `elapsed` (entiers) du run ; fenêtre centrée sur la seconde t :
    # échantillons [25 t - 112, 25 t + 138), soit le milieu de la seconde t au centre de la fenêtre
    raw = pd.read_csv(os.path.join(RAW_DIR, f'{run_id}_mov.csv'), usecols=AXES, engine='pyarrow').to_numpy(np.float64)
    first = 13                                    # 25 t - 112 = 13 + 25 (t - 5)
    out = {k: np.zeros(0) for k in BANDES}
    if len(raw) >= first + WIN:
        tot = {k: 0 for k in BANDES}
        for a in range(3):
            v = sliding_window_view(raw[first:, a], WIN)[::FS]             # fenêtre m <-> seconde t = m + 5
            bp = {k: [] for k in BANDES}
            for i in range(0, len(v), 20000):                              # par paquets (mémoire)
                r = band_powers(v[i:i + 20000])
                for k in BANDES:
                    bp[k].append(r[k])
            for k in BANDES:
                tot[k] = tot[k] + np.concatenate(bp[k])
        out = tot
    t = np.asarray(elapsed, dtype=np.int64)
    m = t - 5
    res = pd.DataFrame({'run_id': run_id, 'elapsed_s': elapsed})
    for k in BANDES:
        arr = np.asarray(out[k])
        ok = (m >= 0) & (m < len(arr))
        col = np.full(len(t), np.nan, dtype=np.float32)
        col[ok] = arr[m[ok]]
        res[k] = col
    return res


# test sur un run : comparaison avec scipy.signal.welch
from scipy.signal import welch
RUN_CONTROLE = os.path.basename(sorted(glob.glob(os.path.join(RAW_DIR, '*_mov.csv')))[0])[:-len('_mov.csv')]
r0 = pd.read_csv(os.path.join(RAW_DIR, f'{RUN_CONTROLE}_mov.csv'), usecols=AXES, nrows=5000).to_numpy(float)
t0 = 100
w = r0[25 * t0 - 112: 25 * t0 + 138]
ref = 0
for a in range(3):
    f, p = welch(w[:, a], fs=FS, window='hann', nperseg=SEG, noverlap=SEG - STEP, detrend='constant')
    ref += p[(f >= 7) & (f <= 12.5)].sum() * (f[1] - f[0])
mine = cl_fourier_run(RUN_CONTROLE, np.arange(95, 105))['CL_Fourier_7_125'].iloc[5]
print(f'contrôle seconde {t0} : scipy {ref:.6e}  |  calcul vectorisé {mine:.6e}')
assert abs(ref - mine) / ref < 1e-4

# ## Calcul pour les 32 patients (parallélisé par run)

jobs = []
for path in sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet'))):
    pid = os.path.basename(path)[:-8]
    d = pd.read_parquet(path, columns=['run_id', 'elapsed_s'])
    for run_id, g in d.groupby('run_id', observed=True):
        jobs.append((pid, str(run_id), g['elapsed_s'].values))
print(f'{len(jobs)} runs à traiter')
t0 = time.time()
results = {}
with ProcessPoolExecutor(N_WORKERS) as ex:
    futs = {ex.submit(cl_fourier_run, r, e): (pid, r) for pid, r, e in jobs}
    for i, fu in enumerate(futs):
        pid, r = futs[fu]
        results.setdefault(pid, []).append(fu.result())
        if (i + 1) % 50 == 0:
            print(f'{i + 1}/{len(jobs)} runs ({time.time() - t0:.0f}s)', flush=True)
for pid, parts in results.items():
    pd.concat(parts, ignore_index=True).to_parquet(os.path.join(OUT_DIR, f'{pid}.parquet'), index=False)
print(f'Terminé en {time.time() - t0:.0f}s')

# ## Contrôle de l'alignement temporel
# 
# On compare `CL_Fourier_2_7` à la feature existante `F_custom_0_5` + `F_custom_5_8` (bandes voisines, même fenêtre de 10 s) :
# la corrélation de rang doit être forte et maximale à un décalage de 0 s si les deux sont datées de la même façon.

# ## Ajout des 2 colonnes aux parquets `data_with_seizure_id`

def h(df):
    return hashlib.md5(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()

for path in sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet'))):
    pid = os.path.basename(path)[:-8]
    df = pd.read_parquet(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith('CL_Fourier')])      # idempotent
    before = h(df)
    c = pd.read_parquet(os.path.join(OUT_DIR, f'{pid}.parquet')).astype({'run_id': str})
    m = df[['run_id', 'elapsed_s']].astype({'run_id': str}).merge(c, on=['run_id', 'elapsed_s'], how='left', validate='one_to_one')
    for k in BANDES:
        df[k] = m[k].values
    assert h(df.drop(columns=list(BANDES))) == before, pid           # colonnes existantes inchangées
    tmp = path + '.tmp'
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    print(f"{pid} : ajouté ({df['CL_Fourier_2_7'].isna().mean():.2%} de NaN)", flush=True)
