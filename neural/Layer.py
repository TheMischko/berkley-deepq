class Layer(object):

    def __init__(self):
        super(Layer, self).__init__()
        self.params = {}
        self.training = True
        self._cache = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self):
        return (param for name, param in self.params.items())

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __getattr__(self, key):
        return self.__dict__.get(key, self.params[key])