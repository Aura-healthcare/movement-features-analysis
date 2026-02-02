
import numpy as np
import pandas as pd
import features_functions as ff
import argparse
from datetime import datetime
import os

# preprocessing and pandas stuff

def preprocess_datas_timeline(datas):

    l_time = [t[0:-5] for t in datas['time']]
    l_time = [datetime.strptime(t, '%Y-%m-%d_%H:%M:%S.%f').timestamp() * 1000 for t in l_time]
    
    t0 = l_time[0]
    datas['timeline_sec'] = [(i - t0) / 1000 for i in l_time]
    return(datas)

def return_euclidean_norm(df):
    ''' 
    return df with new column norme
    require columns x,y,z 
    '''
    x = np.asarray(df['x'])
    y = np.asarray(df['y'])
    z = np.asarray(df['z'])
    df['norme'] = np.linalg.norm([x,y,z],2,axis = 0) 

    return(df)

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

################################### Get dataframe with features ###########################################################################

def get_dataframe_features(datas,window_size_sec,fs,list_features_type):
    ''' 
    list_features_type : ['temporal_features','frequential_features','multiple_parameters_frequential_features','fourier_spectrum_features']
    '''
    
    df_features = pd.DataFrame() 
    dict_args_value = {'fs' : fs}

    ''' 
    inside macro parameters
    '''
    datas = preprocess_datas_timeline(datas)
    datas = return_euclidean_norm(datas)

    file_duration_sec = round(datas['timeline_sec'].iloc[-1],0)
    nb_row_features  = int((file_duration_sec // window_size_sec * 2) - 1 )
    print('file_duration_sec = ',file_duration_sec)
    print('nb_row_features to calculate = ', nb_row_features)

    nb_uncalculated,nb_zeros = 0,0

    for row_number in range(nb_row_features):

        if row_number % 2000 == 0:
            print("row_number = ", row_number)

        # je récupère le signal pour chaque ligne de feature : df_feature_calcul_window
        begin_sec = 5 * row_number
        df_window = get_df_window(datas,begin_sec,window_size_sec)
        
        if len(df_window) < 50:
            nb_uncalculated += 1
            continue

        df_window_zero = df_window.loc[df_window['norme']==0]
        len_zero = len(df_window_zero) 
        
        if len_zero > 40:
            continue

        dict_args_value['x'] = df_window['x']
        dict_args_value['y'] = df_window['y']
        dict_args_value['z'] = df_window['z']
        dict_args_value['norme'] = df_window['norme']
        all_features = dict()
        

        # Je calcul les features dans dict features, la fréquence ne change pas seul le signal
        for axe in ['x','y','z','norme']:
            
            dict_args_value['signal'] = dict_args_value[axe]
            
            if 'temporal_features' in list_features_type:
                dict_temporal_features = calcul_temporal_features(dict_args_value)
                dict_temporal_features = update_feature_name_with_axe(dict_temporal_features,axe)
                all_features.update(dict_temporal_features)

            if 'frequential_features' in list_features_type:
                dict_frequential_features = calcul_frequential_features(dict_args_value)
                dict_frequential_features = update_feature_name_with_axe(dict_frequential_features,axe)
                all_features.update(dict_frequential_features)

            if 'multiple_parameters_frequential_features' in list_features_type:
                dict_multiple_parameters_frequential_features = calcul_frequential_features_with_parameters(dict_args_value)
                dict_multiple_parameters_frequential_features = update_feature_name_with_axe(dict_multiple_parameters_frequential_features,axe)
                all_features.update(dict_multiple_parameters_frequential_features)

        if 'fourier_spectrum_features' in list_features_type:
            dict_fourier_spectrum_features = calcul_fourier_features(dict_args_value,fs)
            all_features.update(dict_fourier_spectrum_features)
            all_features['time'] = df_window['time'].iloc[0]

        df_features = df_features._append(all_features,ignore_index=True)
        

    print('nb_uncalculated = ',nb_uncalculated)

    return(df_features)

##############################################################Parsing and extraction #######################################################################################################

default_list_features_type = ['temporal_features','fourier_spectrum_features','frequential_features'] # ,'multiple_parameters_frequential_features'] # ['temporal_features','fourier_spectrum_features'] #,'frequential_features','multiple_parameters_frequential_features','fourier_spectrum_features']


# parser.add_argument("datas_path", const = 1, help="absolute or relaive path to your datas",type = str)
# default = datas_path, 
# parser.add_argument("window_size_sec", default = 10, help="the window size in second on which you calculate your features",type=int)
# parser.add_argument("features_type", nargs='?', const = 1, default = 'all',help="list of features type, default = all")
# if args.features_type == 'all':
#     list_features_type = default_list_features_type
# else:
#     list_features_type = args.features_type

if __name__ == '__main__':
    
    window_size_sec = 10 # args.window_size_sec

    parser = argparse.ArgumentParser(description="Chemins pour charger et sauvegarder les fichiers.")

    parser.add_argument('-i',
        "--datas_path",
        required=True,
        type=str,
        help="Chemin vers le fichier datas"
    )

    parser.add_argument('-o',
        "--save_path",
        required=True,
        type=str,
        help="Chemin vers la sauvegarde"
    )

    args = parser.parse_args()
    datas_path = args.datas_path
    save_path = args.save_path
    datas = pd.read_csv(datas_path ,index_col = 0)

    print('file : ', datas_path)
    print('shape datas = ', np.shape(datas))
    datas.rename({'acc_x' : 'x','acc_y' : 'y','acc_z' : 'z'},axis = 1, inplace=True)
    datas[['x', 'y','z']] = datas[['x', 'y','z']].fillna(value=0)
    fs = 50
    datas['time'] = datas.index

    df_results = get_dataframe_features(datas,window_size_sec,fs,default_list_features_type)
    print('nb lignes features calculatesd = ', np.shape(df_results)[0])
    
    df_results.to_csv(save_path)
    print(save_path, ' saved')
