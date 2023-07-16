
import numpy as np
import pandas as pd
# import scipy.signal
import features_functions as ff


dict_features_argument = dict()

def calcul_temporal_features(dict_args_value,dict_time_domain_mvt_features = ff.dict_time_domain_mvt_features,
                             dict_time_domain_arguments = ff.dict_time_domain_arguments):

    dict_features_value = dict()

    for feature,fonction in ff.dict_time_domain_mvt_features.items():
        arg_name = dict_time_domain_arguments[feature]
        arg_value = dict_args_value[arg_name]
        feature_value = fonction(arg_value)
        dict_features_value[feature] = feature_value
    
    return(dict_features_value)

def calcul_frequential_features(dict_args_value,dict_frequential_mvt_features = ff.dict_frequential_mvt_features):

    signal = dict_args_value['signal']
    fs =  dict_args_value['fs']
    dict_features_value = dict()

    for feature,fonction in dict_frequential_mvt_features.items():
        feature_value = fonction(signal,fs)
        dict_features_value[feature] = feature_value
    
    return(dict_features_value)

def calcul_fourier_features(dict_args_value,freq_acq):
    
    signal_3d = dict_args_value['signal_3d']

    x = signal_3d[0]
    y = signal_3d[1]
    z = signal_3d[2]

    fft = ff.calcul_3d_fourier(x,y,z)
    df_freq = ff.get_df_freq(fft,20)
    dict_features = ff.get_spectrum_features(df_freq,freq_acq)

    return(dict_features)