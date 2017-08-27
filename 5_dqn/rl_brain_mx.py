import numpy as np 
import mxnet as mx 

np.random.seed(1)

class DeepQNetwork:
    def __init__(self,
            n_actions,
            n_features,
            learning_rate=0.01)