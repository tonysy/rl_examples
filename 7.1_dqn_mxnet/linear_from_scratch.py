import mxnet as mx
from mxnet import nd, autograd
mx.random.seed(1)

num_inputs = 2
num_outputs = 1
num_examples = 10000

X = nd.random_normal(shape=(num_examples, num_inputs))
