
class Loader(object):
    def __init__(self):
        super(Loader, self).__init__()

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
