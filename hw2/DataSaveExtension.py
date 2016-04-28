__author__ = 'Steffen'

import numpy as np
from blocks.extensions import SimpleExtension


class SaveWeights(SimpleExtension):
    def __init__(self, layers, prefixes, **kwargs):
        kwargs.setdefault("after_epoch", True)
        super(SaveWeights, self).__init__(**kwargs)
        self.step = 1
        self.layers = layers
        self.prefixes = prefixes

    def do(self, callback_name, *args):
        for i in xrange(len(self.layers)):
            filename = "%s_%d.npy" % (self.prefixes[i], self.step)
            np.save(filename, self.layers[i].parameters[0].get_value())
        self.step += 1