
import numpy as np
import pandas as pd
# import scipy.signal
import features_functions as ff
import argparse


# preprocessing and pandas stuff
def preprocess_datas_timeline(datas):
    t0 = datas.iloc[0]['timeline']
    datas['timeline_sec'] = [(i - t0) / 1000 for i in datas.timeline]
    return(datas)

def return_euclidean_norm(df):
    ''' 
    return df with new column norm
    require columns x,y,z 
    '''
    x = np.asarray(df['x'])
    y = np.asarray(df['y'])
    z = np.asarray(df['z'])
    df['norm'] = np.linalg.norm([x,y,z],2,axis = 0) 

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

def compute_temporal_features(dict_args_value,dict_time_domain_mvt_features = ff.dict_time_domain_mvt_features,
                             dict_time_domain_arguments = ff.dict_time_domain_arguments):

    dict_features_value = dict()

    for feature,fonction in ff.dict_time_domain_mvt_features.items():
        arg_name = dict_time_domain_arguments[feature]
        arg_value = dict_args_value[arg_name]
        feature_value = fonction(arg_value)
        dict_features_value[feature] = feature_value
    
    return(dict_features_value)

##############################################################################################################

def compute_frequential_features(dict_args_value,dict_frequential_mvt_features = ff.dict_frequential_mvt_features):

    signal = dict_args_value['signal']
    fs =  dict_args_value['fs']
    dict_features_value = dict()

    for feature,fonction in dict_frequential_mvt_features.items():
        feature_value = fonction(signal,fs)
        dict_features_value[feature] = feature_value
    
    return(dict_features_value)

############################################################################################################## 

def compute_fourier_features(dict_args_value,freq_acq):
    
    x = dict_args_value['x']
    y = dict_args_value['y']
    z = dict_args_value['z']

    fft = ff.compute_3d_fourier(x,y,z)
    df_freq = ff.get_df_freq(fft,freq_acq)
    dict_features = ff.get_spectrum_features(df_freq,freq_acq)

    return(dict_features)

dict_features_value = dict()

##############################################################################################################

def compute_frequential_features_with_parameters(dict_args_value,dict_frequential_mvt_multiple_features = ff.dict_frequential_mvt_multiple_features):
    
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

    for row_number in range(nb_row_features):
        # je récupère le signal pour chaque ligne de feature : df_feature_compute_window
        begin_sec = 5 * row_number
        df_window = get_df_window(datas,begin_sec,window_size_sec)

        dict_args_value['x'] = df_window['x']
        dict_args_value['y'] = df_window['y']
        dict_args_value['z'] = df_window['z']
        dict_args_value['norm'] = df_window['norm']
        all_features = dict()
        

        # Je calcul les features dans dict features, la fréquence ne change pas seul le signal
        for axe in ['x','y','z','norm']:
            
            dict_args_value['signal'] = dict_args_value[axe]
            
            if 'temporal_features' in list_features_type:
                dict_temporal_features = compute_temporal_features(dict_args_value)
                dict_temporal_features = update_feature_name_with_axe(dict_temporal_features,axe)
                all_features.update(dict_temporal_features)

            if 'frequential_features' in list_features_type:
                dict_frequential_features = compute_frequential_features(dict_args_value)
                dict_frequential_features = update_feature_name_with_axe(dict_frequential_features,axe)
                all_features.update(dict_frequential_features)

            if 'multiple_parameters_frequential_features' in list_features_type:
                dict_multiple_parameters_frequential_features = compute_frequential_features_with_parameters(dict_args_value)
                dict_multiple_parameters_frequential_features = update_feature_name_with_axe(dict_multiple_parameters_frequential_features,axe)
                all_features.update(dict_multiple_parameters_frequential_features)

        if 'fourier_spectrum_features' in list_features_type:
            dict_fourier_spectrum_features = compute_fourier_features(dict_args_value,fs)
            all_features.update(dict_fourier_spectrum_features)

        df_features = df_features.append(all_features,ignore_index=True)
    
    return(df_features)

##############################################################Parsing and extraction #######################################################################################################

default_list_features_type = ['temporal_features','frequential_features','multiple_parameters_frequential_features','fourier_spectrum_features']
dir_datas = '../datas/'
datas_test = 'datas_test_feature_extraction.csv'



parser = argparse.ArgumentParser()

parser.add_argument("window_size_sec", help="the window size in second on which you calculate your features",type=int)
parser.add_argument("datas_path",nargs='?', const = 1, default = dir_datas + datas_test, help="absolute or relaive path to your datas",type = str)
parser.add_argument("features_type", nargs='?', const = 1, default = 'all',help="list of features type, default = all")

args = parser.parse_args()

if __name__ == '__main__':
    
    window_size_sec = args.window_size_sec
    datas_path = args.datas_path

    if args.features_type == 'all':
        list_features_type = default_list_features_type
    else:
        list_features_type = args.features_type

    datas = pd.read_csv(datas_path,index_col = 0)
    fs = 45
    df = get_dataframe_features(datas,window_size_sec,fs,list_features_type)
    print('nb lignes features calculatesd = ', np.shape(df)[0])
    print('nb features per lignes = ', np.shape(df)[1])  

    df.to_csv(dir_datas + 'features_{}_ws_{}.csv'.format(datas_test,window_size_sec))