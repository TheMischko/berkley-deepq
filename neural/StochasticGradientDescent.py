
class StochasticGradientDescent(object):

    def __init__(self, params, learning_rate=1e-3, weight_decay_l2=1e-3):
        """
        params = slovni mapujici jmena parametru na jejich hodnoty
        """
        super().__init__()
        self.params = params
        self.learning_rate = learning_rate
        self.weight_decay_l2 = weight_decay_l2

    def step(self, grads):
        """
        Updatuje parametry gradienty specifikovanymi argumentem `grads`

        grads ... slovnik mapujici jmena parametru na jejich gradienty
        """

        for name, grad in grads.items():
            # pokud je nastavena regularizace, pak upravime gradient
            if self.weight_decay_l2 > 0:
                grad = grad + 2. * self.weight_decay_l2 * self.params[name]

            # stochastic gradient descent update
            self.params[name] -= self.learning_rate * grad
