import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


class Stats(object):

    def __init__(self, smooth_coef=0.99):
        super().__init__()

        self._epochs = []
        self._running_avg = {}
        self.smooth_coef = smooth_coef

    def __len__(self):
        return len(self._epochs)

    def append_batch_stats(self, subset, **stats):
        if subset not in self._epochs[-1]:
            self._epochs[-1][subset] = {}
        for k, v in stats.items():
            if k not in self._epochs[-1][subset]:
                self._epochs[-1][subset][k] = []
            self._epochs[-1][subset][k].append(v)

        if subset not in self._running_avg:
            self._running_avg[subset] = {}
        for k, v in stats.items():
            self._running_avg[subset][k] = self.smooth_coef * self._running_avg[subset].get(k, v) + (
                        1. - self.smooth_coef) * v

    def new_epoch(self):
        self._epochs.append({})

    def epoch_average(self, epoch, subset, metric):
        try:
            return np.mean([s for s in self._epochs[epoch][subset][metric]])
        except (KeyError, TypeError):
            return np.nan

    def ravg(self, subset, metric):
        return self._running_avg[subset][metric]

    def summary(self, epoch=-1):
        epoch = self._epochs.index(self._epochs[epoch])

        res = {}
        for subset, stats in self._epochs[epoch].items():
            for metric in stats:
                if metric not in res:
                    res[metric] = {}
                res[metric][subset] = self.epoch_average(epoch, subset, metric)

        return pd.DataFrame(res).rename_axis('Epoch {:02d}'.format(epoch + 1), axis='columns')

    def best_epoch(self, key=None):
        if key is None:
            key = lambda i: self.epoch_average(i, 'valid', 'acc')
        return max(range(len(self)), key=key)

    def best_results(self, key=None):
        return self.summary(self.best_epoch(key=key))

    def _plot(self, xdata, yleft, yright, ax=None, xlabel='batch', ylabels=None, **kwargs):
        """
        vykresli prubeh lossu a prip. do stejneho grafu zakresli i prubeh acc
        """

        if ax is None:
            fig, ax = plt.subplots()

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        styles = ['-', '--', ':']
        labels = ylabels if ylabels is not None else [None] * len(styles)

        def filter_plot_data(x, y):
            ok = ~np.isnan(y)
            return x[ok], y[ok]

        if yleft is not None:
            for i, ydata in enumerate(yleft):
                x, y = filter_plot_data(xdata, np.array(ydata))
                ax.plot(x, y, color=colors[0], linestyle=styles[i], label=labels[i], **kwargs)
            ax.set_ylabel('loss', color=colors[0])
            ax.tick_params(axis='y', colors=colors[0])

        if yright is not None:
            ax2 = ax.twinx()
            for i, ydata in enumerate(yright):
                x, y = filter_plot_data(xdata, np.array(ydata))
                ax2.plot(x, y, color=colors[1], linestyle=styles[i], label=labels[i], **kwargs)
            ax2.set_ylabel('acc', color=colors[1])
            ax2.tick_params(axis='y', colors=colors[1])

        ax.set_xlabel(xlabel)
        ax.figure.tight_layout()
        if ylabels is not None:
            ax.legend()

    def plot_by_batch(self, ax=None, subset='train', left_metric='loss', right_metric='acc', block_len=1):
        if ax is None:
            fig, ax = plt.subplots()

        yleft = None
        if left_metric is not None:
            yleft = [v for i, ep in enumerate(self._epochs) for v in ep[subset][left_metric]]
        yright = None
        if right_metric is not None:
            yright = [v for i, ep in enumerate(self._epochs) for v in ep[subset][right_metric]]
        len_fn = lambda x: len(x) if x is not None else 0
        xdata = 1 + np.arange(max(len_fn(yleft), len_fn(yright)))

        if block_len is not None:
            if yleft is not None:
                yleft = [np.mean(np.reshape(yleft[:block_len * (len(yleft) // block_len)], (-1, block_len)), axis=1)]
            if yright is not None:
                yright = [np.mean(np.reshape(yright[:block_len * (len(yright) // block_len)], (-1, block_len)), axis=1)]
            xdata = xdata[len(xdata) % block_len:: block_len]

        self._plot(xdata, yleft, yright, ax=ax, xlabel='batch')

    def plot_by_epoch(self, ax=None, subsets=('train', 'valid'), left_metric='loss', right_metric='acc'):
        if ax is None:
            fig, ax = plt.subplots()

        xdata = 1 + np.arange(len(self))
        yleft, yright = [], []

        for i, ss in enumerate(subsets):
            yleft.append([])
            yright.append([])
            for j, ep in enumerate(self._epochs):
                yleft[-1].append(self.epoch_average(j, ss, left_metric))
                yright[-1].append(self.epoch_average(j, ss, right_metric))

        self._plot(xdata, yleft, yright, ax=ax, xlabel='epoch', ylabels=subsets)

