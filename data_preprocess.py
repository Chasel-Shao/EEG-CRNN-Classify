import numpy as np
from keras.utils import to_categorical
import tensorflow as tf
import os
from utils import *

def load_data():
    ret = dict()
    X_train_valid = np.load("data/X_train_valid.npy")
    y_train_valid = np.load("data/y_train_valid.npy")
    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")
    person_train_valid = np.load("data/person_train_valid.npy")
    person_test = np.load("data/person_test.npy")
    
    ret['X_test'] = X_test
    ret['y_test'] = y_test
    ret['X_train_valid'] = X_train_valid
    ret['y_train_valid'] = y_train_valid
    ret['person_train_valid'] = person_train_valid
    ret['person_test'] = person_test
    return ret
    

def data_prep(X, y, sub_sample, average, time=500):
    
    total_X = None
    total_y = None
    
    # Trimming the data (sample,22,1000) -> (sample,22,500)
    X = X[:,:,0:time]
    
    if average > 1:
        # Maxpooling the data (sample,22,1000) -> (sample,22,500/sub_sample)
        X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, average), axis=3)
        total_X = X_max
        total_y = y
        # Averaging 
        X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, average),axis=3)
        
        # print("average vstack: total_X.shape {}, X_average.shape {}".format(total_X.shape, X_average.shape))
        total_X = np.vstack((total_X, X_average))
        total_y = np.hstack((total_y, y))
        
    # Subsampling
    if sub_sample > 1:
        for i in range(sub_sample):
            X_subsample = X[:, :, i::sub_sample]
            
            if total_X is None:
                total_X = X_subsample
            else:
                # print("sub_sample vstack: total_X.shape {}, X_subsample.shape {}".format(total_X.shape, X_subsample.shape))
                total_X = np.vstack((total_X, X_subsample))
            
            if total_y is None:
                total_y = y
            else:
                total_y = np.hstack((total_y, y))
    
    if total_X is None:
        total_X = X
    if total_y is None:
        total_y = y
    
    return total_X, total_y

def add_gaussian_noise(data, noise_level = 0.5):
    """
    向数据添加高斯噪声。
    :param data: 标准化后的EEG数据，形状为(sample, electrodes, time_series)
    :param noise_level: 噪声的标准差，决定了噪声的强度
    :return: 添加了高斯噪声的EEG数据
    """
    noise = np.random.normal(0, noise_level, data.shape)
    noisy_data = data + noise
    return noisy_data

def z_score_normalization(eeg_data):
    """
    对EEG数据应用Z得分标准化。
    :param eeg_data: EEG数据数组，形状为(sample, electrodes, time_series)
    :return: 标准化后的EEG数据
    """
    mean = np.mean(eeg_data, axis=2, keepdims=True)
    std = np.std(eeg_data, axis=2, keepdims=True)
    normalized_data = (eeg_data - mean) / std
    return normalized_data

def load_prep_data(time=500, debug=False, pooling = True, subsample = 2, average = 2, normalization = False, noise_level = 0.5, windowing = False, onehot=True):

    ## Loading the dataset
    data_dict = load_data()
    X_train_valid = data_dict["X_train_valid"]
    y_train_valid = data_dict["y_train_valid"]
    X_test = data_dict["X_test"]
    y_test = data_dict["y_test"]
    person_train_valid = data_dict["person_train_valid"]
    person_test = data_dict["person_test"]
    
    ret = dict()
    
    if normalization:
        X_train_valid = z_score_normalization(X_train_valid)
        
    if windowing:
        X_train_valid, y_train_valid = get_windowing(X_train_valid, y_train_valid)
        X_test, y_test = get_windowing(X_test, y_test)
    

    ## Adjusting the labels so that 
    # Cue onset left - 0
    # Cue onset right - 1
    # Cue onset foot - 2
    # Cue onset tongue - 3
    y_train_valid -= 769
    y_test -= 769

    ## Preprocessing the dataset
    x_train_valid, y_train_valid = data_prep(X_train_valid, y_train_valid, subsample, average, time)
    x_test, y_test = data_prep(X_test, y_test, subsample, average, time)
    
    onehot = True
    if onehot:
        # Converting the labels to categorical variables for multiclass classification
        y_train_valid = to_categorical(y_train_valid, 4)
        y_test = to_categorical(y_test, 4)
    
    if noise_level > 0:
        x_train_valid = add_gaussian_noise(x_train_valid, noise_level)
        
    ## Random splitting and reshaping the data
    # First generating the training and validation indices using random splitting
    total_num = x_train_valid.shape[0]
    valid_num = int(total_num * 0.15)
    train_num = total_num - valid_num
    ind_valid = np.random.choice(total_num, valid_num, replace=False)
    ind_train = np.array(list(set(range(total_num)).difference(set(ind_valid))))

    # Creating the training and validation sets using the generated indices
    (x_train, x_valid) = x_train_valid[ind_train], x_train_valid[ind_valid] 
    (y_train, y_valid) = y_train_valid[ind_train], y_train_valid[ind_valid]
    
    if debug:
        print('Shape of x_train set:',x_train.shape)
        print('Shape of y_train labels:',y_train.shape)
        print('Shape of x_valid set:',x_valid.shape)
        print('Shape of y_valid labels:',y_valid.shape)
        print('Shape of x_test set:',x_test.shape)
        print('Shape of y_test labels:',y_test.shape)
        print('Shape of x_train_valid set:',x_train_valid.shape)
        print('Shape of y_train_valid labels:',y_train_valid.shape)
    
    ret['x_train'] = x_train
    ret['y_train'] = y_train
    ret['x_valid'] = x_valid
    ret['y_valid'] = y_valid
    ret['x_test'] = x_test
    ret['y_test'] = y_test
    ret['person_train_valid'] = person_train_valid
    ret['person_test'] = person_test
    
    ret['X_train_valid'] = x_train_valid
    ret['y_train_valid'] = y_train_valid
    return ret

def get_subject_by_index(data_dict, subject_index):
    if subject_index is None or (subject_index is not None and (subject_index > 8 or subject_index < 0)):
        raise Exception("subject_index muste be 0 to 8")
        
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    X_train_valid = data_dict['X_train_valid']
    y_train_valid = data_dict['y_train_valid']
    person_train_valid = data_dict['person_train_valid']
    person_test = data_dict['person_test']
    
    sub_1_train_valid_idx = np.where(person_train_valid.ravel() == subject_index)
    sub_1_test_idx = np.where(person_test.ravel() == subject_index)

    # print(format(sub_1_train_valid_idx))
    print(format(X_train_valid.shape))
    X_train_valid = X_train_valid[sub_1_train_valid_idx]
    y_train_valid = y_train_valid[sub_1_train_valid_idx]

    X_test = X_test[sub_1_test_idx]
    y_test = y_test[sub_1_test_idx]
    
    data = {
    'X_train_valid':X_train_valid,
    'y_train_valid':y_train_valid,
    'X_test':X_test,
    'y_test':y_test,
    }
    
    return data


def get_windowing(X, y, window_size=500, step=100):
    X_window = []
    y_window = []
    for i, x_data in enumerate(X):
        steps = (1000 - window_size) // step + 1
        for j in range(steps):
            new_x_data = x_data[:,j*step:j*step+window_size]
            X_window.append(new_x_data)
            y_window.append(y[i])
    X_window = np.array(X_window)
    y_window = np.array(y_window)
    return X_window, y_window


def test_CNN4LayerGRU(aug_data, model):
    x_train = aug_data['x_train']
    y_train = aug_data['y_train']
    x_valid = aug_data['x_valid']
    y_valid = aug_data['y_valid']
    x_test = aug_data['x_test']
    y_test = aug_data['y_test']
    person_train_valid = aug_data['person_train_valid']
    person_test = aug_data['person_test']
    X_train_valid = aug_data['X_train_valid']
    y_train_valid = aug_data['y_train_valid']

    input_shape = x_train.shape
    x_train = tf.transpose( tf.expand_dims(x_train, axis=-1), perm=[0, 2, 3, 1])
    x_valid = tf.transpose( tf.expand_dims(x_valid, axis=-1), perm=[0, 2, 3, 1])
    x_test = tf.transpose(tf.expand_dims(x_test, axis=-1), perm=[0, 2, 3, 1])
    
    print("x_train.shape: {}", format(x_train.shape))
    
    config = {
        # Network
        'num_inputs': x_train.shape[0],
        'input_shape': (input_shape[2], 1, input_shape[1]),
        'epochs': 100,
        'dropout': 0.5,
        'batch_size': 64
    }
    model.build_model(config)
    history = model.train(x_train, y_train, x_valid, y_valid, config, get_workpath('CNN4LayerGRU'))
    
    raw = model.evaluate(x_test, y_test)
    print("Raw Acc result: {}".format(raw[1]))
    
def test_ConvMixGru(aug_data, model):
    x_train = aug_data['x_train']
    y_train = aug_data['y_train']
    x_valid = aug_data['x_valid']
    y_valid = aug_data['y_valid']
    x_test = aug_data['x_test']
    y_test = aug_data['y_test']
    person_train_valid = aug_data['person_train_valid']
    person_test = aug_data['person_test']
    X_train_valid = aug_data['X_train_valid']
    y_train_valid = aug_data['y_train_valid']

    input_shape = x_train.shape
    
    print("x_train.shape: {}", format(x_train.shape))
    
    config = {
    # Network
    'num_inputs': x_train.shape[0],
    'input_shape': (x_train.shape[1],x_train.shape[2],1),
    'epochs': 100,
    'dropout': 0.5,
    'batch_size': 64,
    'l2': 0.03,
    'LSTM': False,
    'lr': 0.001
    }
    model.build_model(config)
    history = model.train(x_train, y_train, x_valid, y_valid, config, get_workpath('ConvMixGru'))
    
    raw = model.evaluate(x_test, y_test)
    print("Raw Acc result: {}".format(raw[1]))
    
def test_VanillaRNN(aug_data, model):
    x_train = aug_data['x_train']
    y_train = aug_data['y_train']
    x_valid = aug_data['x_valid']
    y_valid = aug_data['y_valid']
    x_test = aug_data['x_test']
    y_test = aug_data['y_test']
    person_train_valid = aug_data['person_train_valid']
    person_test = aug_data['person_test']
    X_train_valid = aug_data['X_train_valid']
    y_train_valid = aug_data['y_train_valid']

    input_shape = x_train.shape
    
    print("x_train.shape: {}", format(x_train.shape))
    
    config = {
    # Network
    'num_inputs': x_train.shape[0],
    'input_shape': (x_train.shape[1],x_train.shape[2],1),
    'epochs': 100,
    'dropout': 0.5,
    'batch_size': 64,
    'l2': 0.03,
    'LSTM': False,
    'lr': 0.001
    }
    model.build_model(config)
    history = model.train(x_train, y_train, x_valid, y_valid, config, get_workpath('VanillaRNN'))
    
    raw = model.evaluate(x_test, y_test)
    print("Raw Acc result: {}".format(raw[1]))
    
