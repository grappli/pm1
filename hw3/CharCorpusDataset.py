__author__ = 'Steffen'

from fuel.datasets import Dataset
from sklearn import preprocessing
from fuel.datasets.hdf5 import H5PYDataset
from pandas import get_dummies
from fuel.datasets import IndexableDataset
import numpy
import h5py

class HDF5CharEncoder:

    def __init__(self, input_file, seq_len):
        with open(input_file) as f:
            #prepare dataset from file

            self.chars = list(f.read())
            self.encoder = preprocessing.LabelEncoder()
            self.numeric_data = self.encoder.fit_transform(self.chars)
            self.num_instances = len(self.chars) / seq_len
            self.data = numpy.reshape(self.numeric_data[:seq_len * self.num_instances], (seq_len, self.num_instances))
            self.length = len(self.encoder.classes_)

    def get_dataset(self):
        return IndexableDataset({'x': self.data})

    def write(self, output_file):
        with h5py.File(output_file, mode='w') as f:
            fx = f.create_dataset('x', self.data.shape, dtype='int64')
            fx[...] = self.data
