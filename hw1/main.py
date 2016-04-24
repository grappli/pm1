__author__ = 'Steffen'

from blocks.bricks import MLP, Softmax, Logistic
from blocks.roles import WEIGHT
from blocks.main_loop import MainLoop
from blocks.initialization import IsotropicGaussian, Constant
from blocks.algorithms import GradientDescent, Scale
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.extensions import Printing, FinishAfter
from blocks.extensions.monitoring import TrainingDataMonitoring, DataStreamMonitoring
from blocks_extras.extensions.plot import Plot
from blocks.filter import VariableFilter

from blocks.bricks.cost import BinaryCrossEntropy, MisclassificationRate

from fuel.datasets.hdf5 import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

from theano import tensor

from hinton import hinton

x = tensor.matrix('x')
y = tensor.lmatrix('y')

mlp = MLP(activations=[Logistic(), Softmax()],
          dims=[117, 55, 2],
          weights_init=IsotropicGaussian(),
          biases_init=Constant(0.01))

mlp.initialize()

y_hat = mlp.apply(x)

cost = BinaryCrossEntropy().apply(y, y_hat)

cg = ComputationGraph(cost)

W1, W2 = VariableFilter(roles=[WEIGHT])(cg.variables)
cost = cost + 0.001 * abs(W1).sum() + 0.001 * abs(W2).sum()
cost.name = 'cost'

error_rate = MisclassificationRate().apply(y.argmax(axis=1), y_hat)
error_rate.name = 'error_rate'

algorithm = GradientDescent(cost=cost,
                            parameters=cg.parameters,
                            step_rule=Scale(learning_rate=0.1))

train_set = H5PYDataset('mushrooms.hdf5', which_sets=('train',))
train_stream = DataStream.default_stream(
    train_set, iteration_scheme=SequentialScheme(
        train_set.num_examples, batch_size=128))

test_set = H5PYDataset('mushrooms.hdf5', which_sets=('test',))
test_stream = DataStream.default_stream(
    test_set, iteration_scheme=SequentialScheme(
        test_set.num_examples, batch_size=128))

main = MainLoop(
    model=Model(cost),
    data_stream=train_stream,
    algorithm=algorithm,
    extensions=[
        FinishAfter(after_n_epochs=10),
        Printing(),
        TrainingDataMonitoring([cost, error_rate], after_batch=True, prefix='train'),
        DataStreamMonitoring([cost, error_rate], after_batch=True, data_stream=test_stream, prefix='test'),
        Plot('Train',
             channels=[['train_cost', 'test_cost'], ['train_error_rate', 'test_error_rate']])
    ])

main.run()

hinton(W1.get_value())
hinton(W2.get_value())
