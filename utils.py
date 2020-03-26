import numpy as np
import keras
from keras import backend as K
from keras.datasets import mnist, fashion_mnist
import pickle as pkl
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
import math
import tensorflow as tf
import torch
from directories import *
from pandas import DataFrame
from torch.utils.data import DataLoader
import random


def execution_time(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nExecution time = {:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds)))


################
# data loaders #
################

def data_loaders(dataset_name, batch_size, n_inputs, shuffle=True):
    random.seed(0)
    x_train, y_train, x_test, y_test, input_shape, num_classes, data_format = \
        load_dataset(dataset_name=dataset_name, n_inputs=n_inputs)

    train_loader = DataLoader(dataset=list(zip(x_train, y_train)), batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(dataset=list(zip(x_test, y_test)), batch_size=batch_size, shuffle=shuffle)

    # todo: check shuffling seed on data loaders

    input_size = input_shape[0]*input_shape[1]*input_shape[2]
    output_size = 10

    return train_loader, test_loader, input_size, output_size

def load_fashion_mnist(img_rows=28, img_cols=28, n_inputs=None):
    print("\nLoading fashion mnist.")

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    if n_inputs:
        x_train = x_train[:n_inputs]
        y_train = y_train[:n_inputs]
        x_test = x_test[:n_inputs]
        y_test = y_test[:n_inputs]

    num_classes = 10
    data_format = 'channels_last'

    print('x_train shape:', x_train.shape, '\nx_test shape:', x_test.shape)
    return x_train, y_train, x_test, y_test, input_shape, num_classes, data_format

def load_mnist(img_rows=28, img_cols=28, n_inputs=None):

    print("\nLoading mnist.")

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    if n_inputs:
        x_train = x_train[:n_inputs]
        y_train = y_train[:n_inputs]
        x_test = x_test[:n_inputs]
        y_test = y_test[:n_inputs]

    num_classes = 10
    data_format = 'channels_last'

    print('x_train shape:', x_train.shape, '\nx_test shape:', x_test.shape)
    return x_train, y_train, x_test, y_test, input_shape, num_classes, data_format


def _onehot(integer_labels):
    """Return matrix whose rows are onehot encodings of integers."""
    n_rows = len(integer_labels)
    n_cols = integer_labels.max() + 1
    onehot = np.zeros((n_rows, n_cols), dtype='uint8')
    onehot[np.arange(n_rows), integer_labels] = 1
    return onehot

def onehot_to_labels(y):
    if type(y) is np.ndarray:
        return np.argmax(y, axis=1)
    elif type(y) is torch.Tensor:
        return torch.max(y, 1)[1]

def load_cifar(data_path, n_inputs=None):
    """Return train_data, train_labels, test_data, test_labels
    The shape of data is 32 x 32 x3"""
    x_train = None
    y_train = []

    data_dir=str(data_path)+'cifar-10/'

    for i in range(1, 6):
        data_dic = unpickle(data_dir + "data_batch_{}".format(i))
        if i == 1:
            x_train = data_dic['data']
        else:
            x_train = np.vstack((x_train, data_dic['data']))
        y_train += data_dic['labels']

    test_data_dic = unpickle(data_dir + "test_batch")
    x_test = test_data_dic['data']
    y_test = test_data_dic['labels']

    x_train = x_train.reshape((len(x_train), 3, 32, 32))
    x_train = np.rollaxis(x_train, 1, 4)
    y_train = np.array(y_train)

    x_test = x_test.reshape((len(x_test), 3, 32, 32))
    x_test = np.rollaxis(x_test, 1, 4)
    y_test = np.array(y_test)

    input_shape = x_train.shape[1:]
    num_classes = 10
    data_format = 'channels_first'

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    if n_inputs:
        x_train = x_train[:n_inputs]
        y_train = y_train[:n_inputs]
        x_test = x_test[:n_inputs]
        y_test = y_test[:n_inputs]

    return x_train, _onehot(y_train), x_test, _onehot(y_test), input_shape, num_classes, data_format


def load_dataset(dataset_name, data_path=DATA, n_inputs=None):
    """
    Load dataset.
    :param dataset_name: choose between "mnist" and "cifar"
    :param test: If True only loads the first 100 samples
    """

    if dataset_name == "mnist":
        return load_mnist(n_inputs=n_inputs)
    elif dataset_name == "cifar":
        return load_cifar(data_path=data_path, n_inputs=n_inputs)
    elif dataset_name == "fashion_mnist":
        return load_fashion_mnist(n_inputs=n_inputs)
    else:
        raise ValueError("\nWrong dataset name.")


############
# pickling #
############


def save_to_pickle(data, path, filename):

    print("\nSaving pickle: ", path + filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + filename, 'wb') as f:
        pkl.dump(data, f)


def load_from_pickle(path):

    print("\nLoading from pickle: ", path)
    with open(path, 'rb') as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()

    return data
    



