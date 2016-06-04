from blocks.bricks.recurrent import SimpleRecurrent, GatedRecurrent
from blocks.bricks import Tanh
from blocks.roles import WEIGHT
from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback)
from theano.sandbox.rng_mrg import MRG_RandomStreams
from numpy import random
from blocks.algorithms import Adam
from blocks.main_loop import MainLoop
from ShowOutput import ShowOutput
from RandomSoftmaxEmitter import RandomSoftmaxEmitter
from blocks.initialization import IsotropicGaussian, Constant
from blocks.algorithms import GradientDescent, Scale
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.extensions import Printing, FinishAfter
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks_extras.extensions.plot import Plot
from blocks.extensions.saveload import Checkpoint
from blocks.filter import VariableFilter

from blocks.bricks.cost import BinaryCrossEntropy, MisclassificationRate
import os.path
from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

from blocks.serialization import load
from CharCorpusDataset import HDF5CharEncoder

import theano.tensor


def train():

    if os.path.isfile('trainingdata.tar'):
        with open('trainingdata.tar', 'rb') as f:
            main = load(f)
    else:
        hidden_size = 512
        filename = 'warpeace.hdf5'

        encoder = HDF5CharEncoder('warpeace_input.txt', 1000)
        encoder.write(filename)
        alphabet_len = encoder.length

        x = theano.tensor.lmatrix('x')

        readout = Readout(
            readout_dim=alphabet_len,
            feedback_brick=LookupFeedback(alphabet_len, hidden_size, name='feedback'),
            source_names=['states'],
            emitter=RandomSoftmaxEmitter(),
            name='readout'
        )

        transition = GatedRecurrent(
            activation=Tanh(),
            dim=hidden_size)
        transition.weights_init = IsotropicGaussian(0.01)

        gen = SequenceGenerator(readout=readout,
                                transition=transition,
                                weights_init=IsotropicGaussian(0.01),
                                biases_init=Constant(0),
                                name='sequencegenerator')

        gen.push_initialization_config()
        gen.initialize()

        cost = gen.cost(outputs=x)
        cost.name = 'cost'

        cg = ComputationGraph(cost)

        algorithm = GradientDescent(cost=cost,
                                    parameters=cg.parameters,
                                    step_rule=Scale(0.5))

        train_set = encoder.get_dataset()
        train_stream = DataStream.default_stream(
            train_set, iteration_scheme=SequentialScheme(
                train_set.num_examples, batch_size=128))

        main = MainLoop(
            model=Model(cost),
            data_stream=train_stream,
            algorithm=algorithm,
            extensions=[
                FinishAfter(),
                Printing(),
                Checkpoint('trainingdata.tar', every_n_epochs=10),
                ShowOutput(every_n_epochs=10)
            ])

    main.run()

if __name__ == "__main__":
    train()