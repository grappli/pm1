__author__ = 'Steffen'

from fuel.datasets import Dataset
from nltk.corpus import brown
from nltk.probability import FreqDist
import numpy as np

from theano import tensor


class CorpusDataset(Dataset):

    provides_sources = ('features', 'targets')
    word_dict = dict()
    num_words = 0
    curr_idx = 0
    word_freq = None
    words = []

    def get_vocab_size(self):
        return len([key for key in self.word_freq if self.word_freq[key] > 5]) + 1

    def __init__(self, num_words):
        self.axis_labels = None
        self.num_words = num_words
        self.word_freq = FreqDist(brown.words())
        self.words = [word for word in brown.words() if self.word_freq[word] > 5]
        self.num_examples = len(self.words)

    def get_word_val(self, idx):
        if idx < 0:
            return 0
        try:
            word = self.words[idx]
        except IndexError:
            return 0
        if word in self.word_dict:
            return self.word_dict[word]
        self.curr_idx += 1
        self.word_dict[word] = self.curr_idx
        return self.curr_idx

    def get_data(self, state=None, request=None):
        x = ([self.get_n_prev(idx) + self.get_n_after(idx) for idx in request],
                [self.get_word_val(idx) for idx in request])
        return x

    def get_n_prev(self, idx):
        return [self.get_word_val(b_idx) for b_idx in range(idx - self.num_words, idx)]

    def get_n_after(self, idx):
        return [self.get_word_val(b_idx) for b_idx in range(idx + 1, idx + self.num_words + 1)]
