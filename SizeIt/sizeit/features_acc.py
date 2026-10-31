"""Features de l'accélération (capteur EEG SD, 25 Hz), une ligne toutes les 5 s sur une fenêtre de 10 s.

Calcul repris à l'identique de features_extraction/SZIT_ACC_EEG_SD_features_computation.py (features temporelles,
fréquentielles et spectre de Fourier sur les axes x, y, z et la norme), avec features_functions.py.
"""
import logging

import numpy as np
import pandas as pd

from . import features_functions as ff

CAPTEUR = 'EEG_SD_ACC'
FENETRE_S = 10
PAS_S = 5
FS = 25
TYPES_FEATURES = ['temporal_features', 'frequential_features', 'fourier_spectrum_features']


def _timeline(df):
    df['time'] = pd.to_datetime(df['time'])
    df['timeline_sec'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()
    return df


def _fenetre(df, debut_s, duree_s):
    return df[(df['timeline_sec'] >= debut_s) & (df['timeline_sec'] < debut_s + duree_s)]


def _renommer(dict_features, axe):
    return {f'{axe}_{k}': v for k, v in dict_features.items()}


def _temporelles(args):
    out = {}
    for feature, fonction in ff.dict_time_domain_mvt_features.items():
        out[feature] = fonction(args[ff.dict_time_domain_arguments[feature]])
    return out


def _frequentielles(args):
    return {feature: fonction(args['signal'], args['fs']) for feature, fonction in ff.dict_frequential_mvt_features.items()}


def _fourier(args, fs):
    fft = ff.calcul_3d_fourier(args['x'], args['y'], args['z'])
    return ff.get_spectrum_features(ff.get_df_freq(fft, fs), fs)


def calculer(df):
    """df : accélération renormalisée (colonnes time, EEG SD ACC X/Y/Z, norme_1). Renvoie le tableau des features."""
    df = df.rename({'EEG SD ACC X': 'x', 'EEG SD ACC Y': 'y', 'EEG SD ACC Z': 'z'}, axis=1)
    df[['x', 'y', 'z']] = df[['x', 'y', 'z']].fillna(value=0)
    df = _timeline(df)[['time', 'x', 'y', 'z', 'norme_1', 'timeline_sec']]
    duree = round(df['timeline_sec'].iloc[-1], 0)
    n_lignes = int((duree // FENETRE_S * 2) - 1)
    lignes = []
    args = {'fs': FS}
    t = df['timeline_sec'].to_numpy()
    croissant = bool((np.diff(t) >= 0).all())
    for i in range(n_lignes):
        if croissant:                                  # mêmes lignes que _fenetre, par recherche dichotomique
            a0, a1 = np.searchsorted(t, PAS_S * i, 'left'), np.searchsorted(t, PAS_S * i + FENETRE_S, 'left')
            w = df.iloc[a0:a1]
        else:
            w = _fenetre(df, PAS_S * i, FENETRE_S)
        feats = {}
        for axe in ['x', 'y', 'z', 'norme_1']:
            args[axe] = w[axe].to_numpy(dtype=float)
        for axe in ['x', 'y', 'z', 'norme_1']:
            args['signal'] = args[axe]
            feats.update(_renommer(_temporelles(args), axe))
            feats.update(_renommer(_frequentielles(args), axe))
        feats.update(_fourier(args, FS))
        feats = {f'{CAPTEUR}_{k}': v for k, v in feats.items()}
        feats['time'] = w['time'].iloc[len(w) // 2]
        lignes.append(feats)
    return pd.DataFrame(lignes)


def calculer_fichier(chemin_entree, chemin_sortie):
    """Lit un CSV d'accélération renormalisée, écrit le CSV des features ; renvoie le nombre de lignes (0 si échec)."""
    try:
        df = pd.read_csv(chemin_entree, index_col=0)
        out = calculer(df)
        out.to_csv(chemin_sortie)
        return len(out)
    except Exception as e:                                         # ex. capteur EEG SD absent du fichier
        logging.error(f'{chemin_entree} : {e}')
        return 0
