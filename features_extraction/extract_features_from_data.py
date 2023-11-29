import numpy as np
import pandas as pd
import features_functions as ff
import features_computation as fc
import argparse

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
    df = fc.get_dataframe_features(datas,window_size_sec,fs,list_features_type)
    print('nb lignes features calculatesd = ', np.shape(df)[0])
    print('nb features per lignes = ', np.shape(df)[1])  

    df.to_csv(dir_datas + 'features_{}_ws_{}.csv'.format(datas_test,window_size_sec))

