from scipy.stats import ttest_ind
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from math import sqrt
import os
import copy
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import generate_plots as plots
import math
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import time
import tsfresh
import pywt
from scipy.misc import electrocardiogram
import scipy.signal as signal
from scipy.signal import find_peaks, savgol_filter, butter, lfilter, freqz
import neurokit2 as nk
import heartpy as hp
from heartpy.datautils import rolling_mean, _sliding_window
from scipy.fftpack import fft, ifft
import statistics as stat
import copy
import numpy as np
import pandas as pd
import copy
from keras.layers import *
from keras.models import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import src.parameters as param
import shap
import os
import warnings
warnings.filterwarnings("ignore")

############################################### Kaveesha ####################################################################
def get_person_centered_labelling(data):
    # Person centered data
    data_temp = copy.deepcopy(data)
    data['model_label'] = 2
    for user_id in range(1, 13):
        unique_fatigue_score_list = []
        index_first = data_temp[data['user_id'] == user_id].index[0]
        index_last = data_temp[data['user_id'] == user_id].index[-1] + 1
        for i in range(index_first, index_last):
            if data['window_number'].iloc[i] == 0:
                unique_fatigue_score_list.append(data['mental_fatigue_score'].iloc[i])

        median_mental_fatigue_score = np.median(unique_fatigue_score_list)
        for i in range(index_first, index_last):
            if data['mental_fatigue_score'].iloc[i] <= median_mental_fatigue_score:  # New score
                data.iat[i, data.columns.get_loc('model_label')] = int(0)
            else:
                data.iat[i, data.columns.get_loc('model_label')] = int(1)

    indices_to_remove = data[data['label'] == 'S2'].index.to_list()
    data.drop(indices_to_remove, axis=0, inplace=True)
    data = data.reset_index(drop=True)
    return data


def get_S1vsS3_labelling_data(data):
    data['model_label'] = 2
    indices_to_remove = data[data['label'] == 'S2'].index.to_list()
    data.loc[data['label'] == 'S1', 'model_label'] = int(0)
    data.loc[data['label'] == 'S3', 'model_label'] = int(1)
    data.drop(indices_to_remove, axis=0, inplace=True)
    data = data.reset_index(drop=True)
    return data
############################################### Kaveesha ####################################################################

############################################### Indunil ####################################################################
def divide_into_users(dataset):
    user_list = []
    for i in range (1, 13):
        dataset_copy = copy.deepcopy(dataset)
        index_first = dataset_copy[dataset['user_id'] == i].index[0]
        index_last = dataset_copy[dataset['user_id'] == i].index[-1] + 1

        user = dataset.iloc[index_first:index_last, :].reset_index(drop=True)
        user_list.append(user)
    return user_list

def create_model(hidden_layer_1_size, hidden_layer_2_size, input_layer_size, add_dropout):
    model = Sequential()
    model.add(Dense(hidden_layer_1_size, input_dim=input_layer_size, activation='relu'))
    if add_dropout:
        model.add(Dropout(0.3))
    model.add(Dense(hidden_layer_2_size, activation='relu'))
    if add_dropout:
        model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
############################################### Indunil ####################################################################

############################################### Chalindu ####################################################################
def run_each_device(window_size, device, labelling_type, ml_model, cv_type, model_type):
    if device == 'chest_band':
        filter_chest_band_data(device, window_size, labelling_type, ml_model, cv_type, model_type)

    elif device == 'earable':
        filter_earables_data(device, window_size, labelling_type, ml_model, cv_type, model_type)

    elif device == 'headband':
        filter_headband_data(device, window_size, labelling_type, ml_model, cv_type, model_type)

    elif device == 'wristband':
        filter_wristband_data(device, window_size, labelling_type, ml_model, cv_type, model_type)



def get_file_list(path):
    list_of_files = list()
    for (dirpath, dirnames, filenames) in os.walk(path):
        list_of_files += [os.path.join(dirpath, file) for file in filenames if file.endswith('.csv')]
    return list_of_files
############################################### Chalindu ####################################################################

############################################### Rahal ####################################################################
def remove_labels():
    data_1 = pd.read_csv('../processed_dataset/20/wristband.csv')
    # data_1 = pd.read_csv('output/results_final_after_PCA.csv')
    if 'Unnamed: 0' in data_1: data_1.pop('Unnamed: 0')
    if 'Unnamed: 0.1' in data_1: data_1.pop('Unnamed: 0.1')
    data_1.dropna(inplace=True)
    # indices_to_remove = data_1[data_1['Type'] == 'Hybrid'].index.to_list()
    # data_1.drop(indices_to_remove, axis=0, inplace=True)
    # data = data_1.reset_index(drop=True)

    labels = ['S1', 'S2']
    for label in labels:
        indices_to_remove = data_1[data_1['label'] == label].index.to_list()
        data_1.drop(indices_to_remove, axis=0, inplace=True)
        data_1 = data_1.reset_index(drop=True)

    data_1.to_csv("output/csv_files/results_final_removing_labels.csv")


def split_leave_three_out_cv_hybrid(dataset, train_split_percentage=0.7, seed=1234):
    if param.HYBRID_SPLIT_TYPE == 'random_split':
        return split_leave_three_out_cv_hybrid_random_split(dataset, train_split_percentage, seed)
    elif param.HYBRID_SPLIT_TYPE == 'session_split':
        return split_leave_three_out_cv_hybrid_session_split(dataset, seed)

############################################### Rahal ####################################################################

############################################### Shehan ####################################################################
def save_csv_files(df, name):
    folder_path = 'csv_results'
    if os.path.exists(folder_path):
        print("Folder path \"" + folder_path + "\" exists")
        pass
    else:
        os.makedirs(folder_path)
    df.to_csv(folder_path + '/' + name)

def calculate_cohen_d(mean1, sd1, n1, mean2, sd2, n2, cohen_d_low_high):
    cohen_d = abs(mean1 - mean2) / sqrt(((sd1 * sd1) + (sd2 * sd2))/2)
    sigma_value = sigma_of_cohend(n1, n2, cohen_d)
    if cohen_d_low_high:
        cohen_d_low = cohen_d - 1.96 * sigma_value
        cohen_d_high = cohen_d + 1.96 * sigma_value
        return [cohen_d, cohen_d_low, cohen_d_high]
    return [cohen_d]
############################################### Shehan ####################################################################



def sigma_of_cohend(n1 ,n2 ,cohen_d):
    value = sqrt(((n1 + n2)/(n1 * n2)) + ((cohen_d * cohen_d)/(2 * (n1 + n2))))
    return value


def modify_results_feature_category_final():
    data_1 = pd.read_csv('output/results_feature_category_final.csv')
    if 'Unnamed: 0' in data_1: data_1.pop('Unnamed: 0')
    if 'Unnamed: 0.1' in data_1: data_1.pop('Unnamed: 0.1')

    indices_to_remove = data_1[data_1['Type'] == 'Hybrid'].index.to_list()
    data_1.drop(indices_to_remove, axis=0, inplace=True)
    data = data_1.reset_index(drop=True)

    data.to_csv("output/csv_files/results_feature_category_popu_random_forest.csv")

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a

def remove_noise_PPG(total_data, data):
    columns = ['red', 'ir']
    drop_indices = []
    for column in columns:
        threshold_high = np.mean(total_data[column]) + 3 * np.std(total_data[column])
        threshold_low = np.mean(total_data[column]) - 3 * np.std(total_data[column])

        for data_point in range(data[column].shape[0]):
            if data[column].iloc[data_point] > threshold_high or data[column].iloc[data_point] < threshold_low:
                if data_point not in drop_indices:
                    drop_indices.append(data_point + data.index[0])
    data.drop(drop_indices, inplace=True)
    return data

def split_leave_one_out_cv(dataset):
    splitted_outer_X = []
    splitted_outer_y = []
    for i in range (1, 13):
        dataset_copy = copy.deepcopy(dataset)
        index_first = dataset_copy[dataset['user_id'] == i].index[0]
        index_last = dataset_copy[dataset['user_id'] == i].index[-1] + 1
        outer_test = dataset.iloc[index_first:index_last, :].reset_index(drop=True)
        if index_first == 0:
            outer_train = dataset_copy.iloc[index_last:, :].reset_index(drop=True)
        else:
            outer_train = dataset_copy.iloc[0:index_first , :]
            outer_train = outer_train.append(dataset_copy.iloc[index_last:, :], ignore_index = True)

        outer_test_y = outer_test['model_label']
        outer_test_X = outer_test.drop(columns=['model_label'])

        outer_train_y = outer_train['model_label']
        outer_train_X = outer_train.drop(columns=['model_label'])

        splitted_outer_X.append([outer_train_X, outer_test_X])
        splitted_outer_y.append([outer_train_y, outer_test_y])

    return splitted_outer_X, splitted_outer_y




