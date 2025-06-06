import numpy as np


class NNModule():
    def __init__(self, epoch=0):
        self.weights = {}
        self.biases = {}
        self.epoch = epoch
        self.attention = {}

    def output_parameters(self):
        return self.weights, self.biases
    
    def output_attention_weights(self):
        return self.attention
    
    def store_attention_weights(self, A, layername='ATT'):
        self.attention[layername] = A

    def initialize_parameters(self, matrix_dims, layer_name, bias=True):
        w = np.random.random(matrix_dims)
        # w = np.repeat(w[np.newaxis, :], self.batc_size, axis=0) # repeat weight matrix for each input batch
        self.weights[layer_name] = w
        if bias:
            b = np.random.random((matrix_dims[-1],))
            self.biases[layer_name] = b
        else:
            self.biases[layer_name] = None

    def get_parameters(self, matrix_dims, layer_name, bias=True):
        """
        Create the weight (and bias) matrices for the function/operation that called it
        
        :param matrix_dims (tuple): (input_dim, output_dim) = array's shape. For multi-head attention: (num_heads, input_dim, output_dim)
        :param layer_name (str): name of the layer in question"""
        
        if self.epoch == 0:
            self.initialize_parameters(matrix_dims, layer_name, bias=bias)
        
        return self.weights[layer_name], self.biases[layer_name]
