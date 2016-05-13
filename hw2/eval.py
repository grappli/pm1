__author__ = 'Steffen'

import numpy as np
from CorpusDataset import CorpusDataset
import scipy

def find_closest(word_vec, vecs):
    return np.argsort([scipy.spatial.distance.cosine(row, word_vec) for row in vecs])[0:5]

def find_closest_word(word_vec, vecs):
    x = find_closest(word_vec, vecs)
    return [dataset.word_dict.keys()[dataset.word_dict.values().index(item)] for item in x]

def get_vec(word):
    return vec[dataset.word_dict[word]]

filename = 'proj_10.npy'

dataset = CorpusDataset(0)

vec = np.load('C:\Users\Steffen\PycharmProjects\pm1\hw2\proj_10.npy')
print find_closest_word(get_vec('Paris') - get_vec('France') + get_vec('Germany'), vec)
