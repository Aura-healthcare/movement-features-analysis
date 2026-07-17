# SeizeIt_features_computation 

import os
import pandas as pd
import numpy as np
import time
import sys
sys.path.append("/home/aura-fabien/movement-features-analysis/features_extraction")
import features_computation as fc
import features_functions as ff

import argparse
import logging

logging.basicConfig(
    filename="errors_features_calcul.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# function preprocess SizeIt
def sit_process_timeline(df):
    df["time"] = pd.to_datetime(df["time"])
    df["timeline_sec"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()
    return df

def get_df_window(df,begin_sec,duration_sec,column = 'timeline_sec'):
    ''' 
    return a dataframe from begin_sec to begin_sec + duration_sec on column timeline_sec' 
    '''
    df_window = df[(df['timeline_sec'] >= begin_sec) & (df['timeline_sec'] < begin_sec + duration_sec)]
    return df_window

def update_feature_name_with_axe(dict_features, axe_name): 

        key_list = list(dict_features.keys())
        for key in key_list:
            dict_features[axe_name + '_' + key] = dict_features.pop(key)
        
        return dict_features



##############################################################################################################

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

##############################################################################################################

def calcul_frequential_features(dict_args_value,dict_frequential_mvt_features = ff.dict_frequential_mvt_features):

    signal = dict_args_value['signal']
    fs =  dict_args_value['fs']
    dict_features_value = dict()

    for feature,fonction in dict_frequential_mvt_features.items():
        feature_value = fonction(signal,fs)
        dict_features_value[feature] = feature_value
    
    return(dict_features_value)

############################################################################################################## 

def calcul_fourier_features(dict_args_value,freq_acq):
    
    x = dict_args_value['x']
    y = dict_args_value['y']
    z = dict_args_value['z']

    fft = ff.calcul_3d_fourier(x,y,z)
    df_freq = ff.get_df_freq(fft,freq_acq)
    dict_features = ff.get_spectrum_features(df_freq,freq_acq)

    return(dict_features)

dict_features_value = dict()

##############################################################################################################

def calcul_frequential_features_with_parameters(dict_args_value,dict_frequential_mvt_multiple_features = ff.dict_frequential_mvt_multiple_features):
    
    signal = dict_args_value['signal']
    fs =  dict_args_value['fs']
    dict_features_value = dict()

    for feature,fonction in dict_frequential_mvt_multiple_features.items():

        if feature in ['wavelet_energy','wavelet_std','wavelet_abs_mean']:
            feature_value = fonction(signal)
        
        else:
            feature_value = fonction(signal,fs)

        for i in range(len(feature_value)):
            feature_name = feature + '_' + str(i)
            dict_features_value[feature_name] = feature_value[i]
 
    return dict_features_value 


if __name__ == '__main__':
    
    window_size_sec = 10 # args.window_size_sec
    input_folder = '/home/aura-fabien/SizeIT/Datas/Csv_acc_renormalised/'
    output_folder = '/home/aura-fabien/SizeIT/Datas/Features/parralel/'

    parser = argparse.ArgumentParser(description="Chemins pour charger et sauvegarder les fichiers.")

    parser.add_argument('-i',
        "--datas_filename",
        required=True,
        type=str,
        help="Chemin vers le fichier datas"
    )

    parser.add_argument('-o',
        "--save_filename",
        required=True,
        type=str,
        help="Chemin vers la sauvegarde"
    )

    args = parser.parse_args()
    input_filename = args.datas_filename
    output_filename = args.save_filename

    # datas_path = input_folder + input_filename
    

    if os.path.isabs(args.datas_filename):
        datas_path = args.datas_filename
    else:
        datas_path = os.path.join(IN_DIR, args.datas_filename)

    if os.path.isabs(args.save_filename):
        output_filename = args.save_filename
    else:
        output_filename = os.path.join(IN_DIR, args.save_filename)

    datas = pd.read_csv(datas_path ,index_col = 0)

    l_acc_1 = ['EEG SD ACC X','EEG SD ACC Y', 'EEG SD ACC Z']
    l_acc_2 = ['ECGEMG SD ACC X','ECGEMG SD ACC Y','ECGEMG SD ACC Z']

    # parameters 
    window_size_sec = 10 # args.window_size_sec
    fs = 25
    list_features_type = ['temporal_features','frequential_features','multiple_parameters_frequential_features','fourier_spectrum_features']

    dict_args_value = {'fs' : fs}
    datas.rename({'EEG SD ACC X' : 'x','EEG SD ACC Y' : 'y','EEG SD ACC Z' : 'z'},axis = 1, inplace=True)
    datas[['x', 'y','z']] = datas[['x', 'y','z']].fillna(value=0)
    datas = sit_process_timeline(datas)
    file_duration_sec = round(datas['timeline_sec'].iloc[-1],0)
    nb_row_features  = int((file_duration_sec // window_size_sec * 2) - 1 )

    df_features = pd.DataFrame() 

    for row_number in range(201): # nb_row_features):
        if row_number % 200 ==0:
            print(row_number)
    
        dict_captor_features = dict()
        dict_captor_features['acc_1'] = dict()
        dict_captor_features['acc_2'] = dict()

        for captor in ['acc_1','acc_2']:

            begin_sec = 5 * row_number    
            

            if captor == 'acc_1':
                df_window = get_df_window(datas,begin_sec,window_size_sec)
                norme = 'norme_1'
            
            else: 
                df_window = df_window.drop(columns=["x", "y", "z"])
                df_window = df_window.rename(columns={
                    "ECGEMG SD ACC X": "x",
                    "ECGEMG SD ACC Y": "y",
                    "ECGEMG SD ACC Z": "z"
                })
                norme = 'norme_2'

            dict_args_value['x'] = df_window['x']
            dict_args_value['y'] = df_window['y']
            dict_args_value['z'] = df_window['z']
            dict_args_value[norme] = df_window[norme]

            for axe in ['x','y','z',norme]:
                
                dict_args_value['signal'] = dict_args_value[axe]
                
                if 'temporal_features' in list_features_type:
                    dict_temporal_features = calcul_temporal_features(dict_args_value)
                    dict_temporal_features = update_feature_name_with_axe(dict_temporal_features,axe)
                    dict_captor_features[captor].update(dict_temporal_features)

                if 'frequential_features' in list_features_type:
                    dict_frequential_features = calcul_frequential_features(dict_args_value)
                    dict_frequential_features = update_feature_name_with_axe(dict_frequential_features,axe)
                    dict_captor_features[captor].update(dict_frequential_features)

                if 'multiple_parameters_frequential_features' in list_features_type:
                    dict_multiple_parameters_frequential_features = calcul_frequential_features_with_parameters(dict_args_value)
                    dict_multiple_parameters_frequential_features = update_feature_name_with_axe(dict_multiple_parameters_frequential_features,axe)
                    dict_captor_features[captor].update(dict_multiple_parameters_frequential_features)

            if 'fourier_spectrum_features' in list_features_type:
                dict_fourier_spectrum_features = calcul_fourier_features(dict_args_value,fs)
                dict_captor_features[captor].update(dict_fourier_spectrum_features)
            
            dict_captor_features[captor] = {f"{captor}_{k}": v for k, v in dict_captor_features[captor].items()}
        
        all_features = dict_captor_features['acc_1'] | dict_captor_features['acc_2']
        all_features['time'] = df_window['time'].iloc[0]
        df_features = df_features._append(all_features,ignore_index=True)

    # s'assure que le dossier de sortie existe
    os.makedirs(output_folder, exist_ok=True)

    # récupère juste le nom du fichier, sans path
    base_name = os.path.basename(output_filename)

    # construit le chemin final correctement
    output_path = os.path.join(output_folder, f"feats_{base_name}")

    # sauvegarde
    df_features.to_csv(output_path)

