import numpy as np
import Loader


class BatchLoader(object):

    def __init__(self, X_data, Y_data, batch_size, subset_name, shuffle=True):
        super().__init__()

        self.X_data = X_data
        self.y_data = Y_data
        self.batch_size = batch_size
        self.subset_name = subset_name
        self.shuffle = shuffle

    def __iter__(self):
        # V prvnim cviceni jsme vybirali vzorky do batche nahodne s opakovanim,
        # neboli behem jedne epochy se mohly nektere obrazky opakovat, zatimco
        # na jine se vubec nedostalo. Zde budeme prochazet obrazky opet v nahodnem
        # poradi, ovsem tak, ze za jednu epochu uvidime kazdy obrazek prave jednou.
        # Toho docilime tak, ze data pred pruchodem nahodne zprehazime.
        if self.shuffle:
            perm = np.random.permutation(self.X_data.shape[0])
        else:
            perm = np.arange(self.X_data.shape[0])

        for n in range(len(self)):
            batch_ids = perm[n * self.batch_size: (n + 1) * self.batch_size]
            yield self.X_data[batch_ids], self.y_data[batch_ids]

    def __len__(self):
        return self.X_data.shape[0] // self.batch_size
