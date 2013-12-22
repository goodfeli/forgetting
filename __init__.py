__author__ = "Ian Goodfellow"

import numpy as np

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

def re_initialize(mlp, re_initialize):

    rng = np.random.RandomState([1, 2, 3])

    if not hasattr(mlp, 'monitor_stack'):
        mlp.monitor_stack = [mlp.monitor]
    else:
        mlp.monitor_stack.append(mlp.monitor)
    del mlp.monitor

    for idx in re_initialize:
        layer = mlp.layers[idx]
        for param in layer.get_params():
            if param.ndim == 2:
                value = param.get_value()
                value = rng.uniform(-layer.irange, layer.irange, value.shape)
                param.set_value(value.astype(param.dtype))
            else:
                assert param.ndim == 1
                value = param.get_value()
                value *= 0
                value += layer.bias_hid
                param.set_value(value.astype(param.dtype))

    return mlp

class permute_and_flip(object):

    def __init__(self, flip = True):
        self.flip = flip

    def apply(self, dataset, can_fit=False):

        X = dataset.X
        if X is None:
            print '!!!!!!!!!!!!!!!!!permute_and_flip does nothing because no data!!!!!!!!!!!!!!!'
            return

        rng = np.random.RandomState([17., 35., 19.])
        n = X.shape[1]

        for i in xrange(X.shape[1]):
            j = rng.randint(n)
            tmp = X[:,i].copy()
            X[:,i] = X[:,j].copy()
            X[:,j] = tmp.copy()

        if self.flip:
            dataset.X = 1. - X

class LimitClass(object):
    def __init__(self, include_classes, size = None):
        self.include_classes = include_classes
        self.size = size

    def apply(self, dataset, can_fit = False):
        indexes = []
        for i in xrange(dataset.y.shape[0]):
            if np.argmax(dataset.y[i]) in self.include_classes:
                indexes.append(i)

        dataset.X = dataset.X[indexes]
        y = dataset.y[indexes]

        if self.size is not None:
            index = range(self.size)
            dataset.rng.shuffle(index)
            dataset.X = dataset.X[index]
            y = y[index]

        # make it one_hot again
        one_hot = np.zeros((y.shape[0], len(self.include_classes)), dtype='float32')
        for i in xrange(y.shape[0]):
            one_hot[i, self.include_classes.index(np.argmax(y[i]))] = 1.
        dataset.y = one_hot
        dataset.data_specs[0].components[1].dim = len(self.include_classes)

def concat(datasets):
    Xs = map(lambda x : x.X, datasets)
    ys = map(lambda x: x.y, datasets)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return DenseDesignMatrix(X=X, y=y)
