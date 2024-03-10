import os.path
from os.path import dirname, abspath, exists, join
import os
from os import makedirs, getcwd
import datetime
import numpy as np
import math
from matplotlib import pyplot as plt
import pickle, json

def plot_channel_perclass(data_class, channel_list):
    for i in channel_list:
        channel_10_avg_class0 = np.mean(data_class[0][:, i, :],
                                        axis=0) - np.mean(
            data_class[0][:, i, :])
        channel_10_avg_class1 = np.mean(data_class[1][:, i, :],
                                        axis=0) - np.mean(
            data_class[1][:, i, :])
        channel_10_avg_class2 = np.mean(data_class[2][:, i, :],
                                        axis=0) - np.mean(
            data_class[2][:, i, :])
        channel_10_avg_class3 = np.mean(data_class[3][:, i, :],
                                        axis=0) - np.mean(
            data_class[3][:, i, :])

        plt.figure(figsize=(8, 4))
        plt.plot(channel_10_avg_class0, label='class 0')
        plt.plot(channel_10_avg_class1, label='class 1')
        plt.plot(channel_10_avg_class2, label='class 2')
        plt.plot(channel_10_avg_class3, label='class 3')
        plt.legend(loc="upper right")
        plt.title('Average EEG signal of channel {} for a given class'.format(i))
        plt.xlabel('timestep')
        plt.ylabel('potential')
        plt.show()


def save_data_pickle(dict_obj, save_path):
    path = join(save_path, 'aug_data.pickle')
    print("Saving data pickle...")
    with open(path, 'wb') as fp:
        pickle.dump(dict_obj, fp)
    print("Data pickle Saved.")

def load_data_pickle(load_path):
    path = join(load_path, 'aug_data.pickle')
    print("Loading data pickle...")
    with open(path, 'rb') as fp:
        data_dict = pickle.load(fp)
    print("Data pickle loaded.")
    return data_dict


def load_data_original():
    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")
    person_train_valid = np.load("data/person_train_valid.npy")
    X_train_valid = np.load("data/X_train_valid.npy")
    y_train_valid = np.load("data/y_train_valid.npy")
    person_test = np.load("data/person_test.npy")

    print('Training/Valid data shape: {}'.format(X_train_valid.shape))
    print('Test data shape: {}'.format(X_test.shape))
    print('Training/Valid target shape: {}'.format(y_train_valid.shape))
    print('Test target shape: {}'.format(y_test.shape))
    print('Person train/valid shape: {}'.format(person_train_valid.shape))
    print('Person test shape: {}'.format(person_test.shape))

    return {'X_test': X_test, 'y_test': y_test,
            'X_train_valid': X_train_valid, 'y_train_valid': y_train_valid}


def replace_model_if_better(model_name, new_acc, new_model, config, history=None):
    model_path = get_model_path(model_name)
    best_val_path = join(model_path, 'best_val')
    ensure_dir(best_val_path)
    old_model_path = join(best_val_path, model_name + '.pickle')
    if os.path.isfile(old_model_path):
        print("Old model exists. Comparing performance.")
        f = open(old_model_path, 'rb')
        model_dict = pickle.load(f)
        f.close()
        if new_acc > model_dict['acc']:
            print("New model is better than the old one. Replacing the old model with the new model.")
            os.remove(old_model_path)
            f = open(old_model_path, 'wb')
            model_dict = {'acc': new_acc, 'model': new_model}
            if history is not None:
                model_dict['history'] = history
            pickle.dump(model_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
            folder_name = dirname(old_model_path)
            config_path = join(folder_name, 'config.json')
            with open(config_path, 'w') as fp:
                json.dump(config, fp, indent=4)
            return True
        else:
            print("New model is worse than the old one. Will not update the old model")
            return False
    else:
        print("No existing model in specified path. Saving the new model")
        f = open(old_model_path, 'wb')
        model_dict = dict()
        model_dict['acc'] = new_acc
        model_dict['model'] = new_model
        if history is not None:
            model_dict['history'] = history
        pickle.dump(model_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
        folder_name = dirname(old_model_path)
        config_path = join(folder_name, 'config.json')
        with open(config_path, 'w') as fp:
            json.dump(config, fp, indent=4)
        return True

    
def get_root_path():
    return getcwd()


def get_data_path():
    return join(get_root_path(), 'data')


def get_save_path():
    return join(get_root_path(), 'save')


def ensure_dir(d):
    if not exists(d):
        makedirs(d)
        
def get_model_path(name):
    return join(get_save_path(), name)
        
def get_workpath(name):
    save_path = get_model_path(name)
    time = str(datetime.datetime.now()).replace(' ', '_')
    workpath = join(save_path, time)
    ensure_dir(workpath)
    ensure_dir(save_path)
    return workpath
