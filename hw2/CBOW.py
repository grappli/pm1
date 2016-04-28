__author__ = 'Steffen'

from CorpusDataset import CorpusDataset
from blocks.bricks import Linear, Softmax
from blocks.bricks.cost import CategoricalCrossEntropy
from blocks.graph import ComputationGraph
from blocks.roles import WEIGHT
from blocks.filter import VariableFilter
from theano import tensor
from DataSaveExtension import SaveWeights
from blocks.bricks.lookup import LookupTable
from blocks.algorithms import GradientDescent, Scale
from blocks.initialization import IsotropicGaussian, Constant

from blocks.model import Model
from blocks.extensions import Printing, FinishAfter
from blocks.extensions.monitoring import TrainingDataMonitoring

from blocks.main_loop import MainLoop

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

num_words = 1
dataset = CorpusDataset(num_words)

hidden_size = 100

x = tensor.imatrix('features')
y = tensor.ivector('targets')

v = dataset.get_vocab_size()

input_to_hidden = LookupTable(name='input_to_hidden', length=v, dim=hidden_size)
h = tensor.mean(input_to_hidden.apply(x), axis=1)


hidden_to_output = Linear(name='hidden_to_output', input_dim=hidden_size, output_dim=v)
y_hat = Softmax().apply(hidden_to_output.apply(h))

input_to_hidden.weights_init = hidden_to_output.weights_init = IsotropicGaussian(0.01)
input_to_hidden.biases_init = hidden_to_output.biases_init = Constant(0)

input_to_hidden.initialize()
hidden_to_output.initialize()

cost = CategoricalCrossEntropy().apply(y, y_hat)

cg = ComputationGraph(cost)

algorithm = GradientDescent(cost=cost,
                            parameters=cg.parameters,
                            step_rule=Scale(learning_rate=0.1))

train_stream = DataStream.default_stream(dataset,
                                         iteration_scheme=SequentialScheme(
                                             dataset.num_examples, batch_size=128))

main = MainLoop(
    model=Model(cost),
    data_stream=train_stream,
    algorithm=algorithm,
    extensions=[
        FinishAfter(after_n_epochs=5),
        Printing(),
        TrainingDataMonitoring([cost], after_batch=True, prefix='train'),
        SaveWeights([input_to_hidden], ['proj'])
    ])

main.run()