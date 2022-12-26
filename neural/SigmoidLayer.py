import torch

from Layer import Layer


class SigmoidLayer(Layer):

    def __init__(self):
        super().__init__()

    def forward(self, score):
        hidden = 1 / (1 + torch.exp(-score))

        self._cache = {
            'hidden': hidden
        }

        return hidden

    def backward(self, dhidden):
        h = self._cache['hidden']
        dscore = dhidden * h * (1 - h)
        dparams = {}

        return dscore, dparams