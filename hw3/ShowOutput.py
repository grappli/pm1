__author__ = 'Steffen'

from gen import gen
from blocks.extensions import SimpleExtension

class ShowOutput(SimpleExtension):

    def do(self, which_callback, *args):
        gen()
