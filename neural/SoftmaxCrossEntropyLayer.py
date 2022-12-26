import torch

from Layer import Layer


class SoftmaxCrossEntropyLayer(Layer):

    def __init__(self, average=True):
        """
        average ... if `True`, loss won't be sum but average over whole batch
        """
        super().__init__()
        self.average = average

    def forward(self, scores, targets=None):
        """
        Forward run of this layer.

        inputs:
            scores  ... N x C score matrix where nth row is score for nth input
            targets ... int vector with dimension N where nth item is index of right class of nth input

            If `targets` are `None` then `scores` must be `tuple` containing both inputs.
        """

        if targets is None:
            scores, targets = scores

        # Score normalization
        scores = scores - torch.max(scores, dim=1).values.reshape(-1, 1)
        probs = torch.exp(scores) / torch.sum(torch.exp(scores), dim=1).reshape(-1, 1)
        loss = torch.sum(-torch.log(probs[range(targets.shape[0]), targets]))

        if self.average:
            loss = loss / targets.shape[0]

        self._cache = {
            'probs': probs,
            'targets': targets,
            'scores': scores
        }

        return loss

    def backward(self, dloss=1.):
        """
        Backward run of this layer.

        inputs:
            dloss  ... gradient from layer before this one

        returns:
            dscores ... N x D gradient on input
            grads   ... dictionary mapping name of parameters onto their gradients
            (will be empty because Softmax doesn't have any parameters)
        """

        # Load from cache
        d_scores = self._cache['probs']
        targets = self._cache['targets']

        # Prepare matrix with ones on indexes of right label.
        y_matrix = torch.zeros(targets.shape[0], d_scores.shape[1])
        y_matrix[range(targets.shape[0]), targets] = 1

        # Compute loss
        d_scores = d_scores - y_matrix * dloss
        if self.average:
            d_scores = d_scores / targets.shape[0]

        return d_scores, {}
