"""Fourier utils functions"""

import numpy as np
import pandas as pd

def compute_3d_fourier(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    
    x = x - np.mean(x)
    y = y - np.mean(y)
    z = z - np.mean(z)

    fft_x = np.abs(np.fft.fft(x))
    fft_y = np.abs(np.fft.fft(y))
    fft_z = np.abs(np.fft.fft(z))

    return(np.linalg.norm([fft_x,fft_y,fft_z],2,axis = 0))

def get_df_freq(fft,freq_acq):
    
    ''' 
    Je retourne juste un df avec une colonne avec les fréquences renseignées
    L'intérêt est que c'est insensible au changement de taille d'epoch
    '''

    nb_intervalles = len(fft) / freq_acq
    nb_coef = len(fft)
    col_freq = np.arange(nb_coef) / (nb_intervalles)
    df_freq = pd.DataFrame(data = col_freq,columns=['freq'])
    df_freq['coef'] = fft / (freq_acq * 2)

    return(df_freq.iloc[0:nb_coef // 2])

def get_spectrum_features(df_freq,freq_acq,l_custom_freq = [0,5,8,12,25]):
    
    dict_features = dict()
    
    for i in range(freq_acq // 2):
        feature_name = 'F_' + str(i) + '_' + str(i+1)
        df_loc = df_freq[(df_freq['freq'] >= i) & (df_freq['freq'] < i + 1)]
        feature_value = df_loc.sum()['coef']
        dict_features[feature_name] = feature_value

    for i in range(len(l_custom_freq) - 1):
        low = l_custom_freq[i]
        up = l_custom_freq[i + 1]
        feature_name = 'F_custom_' + str(low) + '_' + str(up)
        df_loc = df_freq[(df_freq['freq'] >= low) & (df_freq['freq'] < up)]
        feature_value = df_loc.sum()['coef']
        dict_features[feature_name] = feature_value
    
    return(dict_features)