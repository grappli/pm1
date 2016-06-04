__author__ = 'Steffen'

from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback)
from theano.sandbox.rng_mrg import MRG_RandomStreams
import random

class RandomSoftmaxEmitter(SoftmaxEmitter):
    def __init__(self):
        SoftmaxEmitter.__init__(self)
        self.theano_rng = MRG_RandomStreams(seed=random.randint(0,100000))
