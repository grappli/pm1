from imagenet import ImagenetModel
import theano
from theano import tensor
import h5py

mat_path = 'imagenet-vgg-verydeep-16.mat'
hdf5_path = '/projects/korpora/mscoco/coco.hdf5'

inet_model = ImagenetModel(mat_path)
data = h5py.File(hdf5_path)

x = tensor.tensor4('input', dtype='float32')

y_hat = x

for layer in inet_model.layers:
    y_hat = layer.apply(y_hat)

func = theano.function([x], y_hat, allow_input_downcast=True)

output = func(data)
