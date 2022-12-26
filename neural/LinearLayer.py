import numpy as np
import torch

from Layer import Layer


class LinearLayer(Layer):
    def __init__(self, input_dim, output_dim, weight_scale=1e-3):
        """
        input_dim    ... input dimension
        output_dim   ... output dimension
        weight_scale ... can be `float` or string with name of the initializing method
        """
        super().__init__()

        self.weight_scale = weight_scale
        self.params = {
            'weights': torch.randn(input_dim, output_dim) / np.sqrt(input_dim),
            'bias': torch.zeros(output_dim)
        }

    def forward(self, inputs):
        """
        Forward run of this layer.

        inputs:
            inputs  ... N x D input matrix with batch size N

        return:
            score   ... N x H output of linear score with batch size N
        """
        w = self.params['weights']
        b = self.params['bias']
        score = (inputs @ w) + b
        self._cache = {'score': score, 'inputs': inputs}
        return score

    def backward(self, dscore):
        """
        Backward run of this layer

        inputs:
            dscore  ... N x H incoming gradient from layer before this one with batch size of N

        return:
            dinputs ... N x D gradient on input, means on `inputs` for method `forward`
            grads   ... dictionary mapping name of parameters onto their gradients
        """

        # load from cache
        f_score = self._cache['score']
        inputs = self._cache['inputs']
        w = self.params['weights']

        # evaluate gradients
        dinputs = dscore @ torch.t(w)
        dweights = torch.t(inputs) @ dscore
        dbias = torch.t(torch.sum(dscore, dim=0))

        return dinputs, {'weights': dweights, 'bias': dbias}
