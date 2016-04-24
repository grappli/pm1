__author__ = 'Steffen'

import h5py
from fuel.datasets.hdf5 import H5PYDataset
from numpy import genfromtxt
import numpy as np
from pandas import get_dummies
import hdf5


def create_hdf5(np_enc_data, np_enc_y, splitpoint, N):

    hdf5name = 'mushrooms.hdf5'
    f = h5py.File(hdf5name, mode='w')

    fx = f.create_dataset('x', np_enc_data.shape, dtype='float32')
    fy = f.create_dataset('y', np_enc_y.shape, dtype='int64')

    fx[...] = np_enc_data
    fy[...] = np_enc_y

    split_dict = {
        'train': {'x': (0,splitpoint), 'y': (0, splitpoint)},
        'test': {'x': (splitpoint, N), 'y': (splitpoint, N)}}

    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    f.flush()
    f.close()


def gen_hdf5():
    data = np.array(genfromtxt('agaricus-lepiota.data', delimiter=',', dtype='c'))
    output_vector = np.array(get_dummies(data.T[0]).values)
    feature_vector = np.zeros((8124,0))
    for column in data.T[1:]:
        feature_vector = np.concatenate((feature_vector, np.array(get_dummies(column).values)), 1)

    hdf5.create_hdf5(feature_vector, output_vector, 4062, 8124)