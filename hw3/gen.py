__author__ = 'Steffen'

from blocks.serialization import load
from blocks.model import ComputationGraph
from CharCorpusDataset import HDF5CharEncoder
from RandomSoftmaxEmitter import RandomSoftmaxEmitter


def gen():
    encoder = HDF5CharEncoder('warpeace_input.txt', 200)

    classes = encoder.encoder.classes_

    seq_len = 100

    with open('trainingdata.tar', 'rb') as f:
        model = load(f).model

    generator = model.top_bricks[0]

    #model.get_parameter_dict()["/sequencegenerator/with_fake_attention/simplerecurrent.initial_state"].set_value()

    f = ComputationGraph(generator.generate(n_steps=1, batch_size=1, iterate=True)).get_theano_function()

    output = ''

    for n in range(seq_len):
        (next_state, next_char, cost) = f()

        output += str(classes[next_char[0][0]])

        #model.get_parameter_dict()["/sequencegenerator/with_fake_attention/simplerecurrent.initial_state"].set_value(next_state[0][0])

    print output

if __name__ == "__main__":
    gen()

