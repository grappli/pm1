__author__ = 'Steffen'

import numpy as np
from CorpusDataset import CorpusDataset
import scipy

def find_closest(word_vec, vecs):
    return np.argmin([scipy.spatial.distance.cosine(row, word_vec) for row in vecs])

def find_closest_word(word_vec, vecs):
    x = find_closest(word_vec, vecs)
    return dataset.word_dict.keys()[dataset.word_dict.values().index(x)]

filename = 'proj_10.npy'

dataset = CorpusDataset(0)

vec = np.load('C:\Users\Steffen\PycharmProjects\pm1\hw2\proj_10.npy')
print find_closest_word(vec[dataset.word_dict['Berlin']] - vec[dataset.word_dict['nice']] + vec[dataset.word_dict['elephant']], vec)
