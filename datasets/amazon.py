__authors__ = "Mehdi Mirza"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Mehdi Mirza"]
__license__ = "3-clause BSD"
__maintainer__ = "Mehdi Mirza"
__email__ = "mirzamom@iro"

import numpy as N
np = N
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils import serial
from theano import config

class AmazonSmall(dense_design_matrix.DenseDesignMatrix):

    valid_sets = ['train', 'test']
    valid_categories = ['kitchen', 'dvd', 'books', 'electronics', 'all']

    def __init__(self, which_set, category, num_feats = 5000,
            shuffle = False,
            one_hot = False,
            start = None,
            stop = None,
            preprocessor = None,
            fit_preprocessor = False,
            fit_test_preprocessor = False):

        self.__dict__.update(locals())
        del self.self

        if which_set not in AmazonSmall.valid_sets:
            raise ValueError("Wrong which_set type, valid options are: {}".\
                    format(AmazonSmall.valid_sets))
        if category not in AmazonSmall.valid_categories:
            raise ValueError("Wrong which_set type, valid options are: {}".\
                    format(AmazonSmall.valid_categories))

        root_path = "${PYLEARN2_DATA_PATH}/multi_domain_sentiment_analysis/" \
                        + "acl_07/numpy-data/"
        path = "{}in-domain-{}-{}-{}-x.npy".format(root_path,
                                                    self.which_set,
                                                    self.category,
                                                    self.num_feats)
        X = serial.load(path)
        X = N.cast[config.floatX](X)
        path = "{}in-domain-{}-{}-{}-y.npy".format(root_path,
                                                    self.which_set,
                                                    self.category,
                                                    self.num_feats)
        y = serial.load(path)

        self.one_hot = one_hot
        if one_hot:
            one_hot = N.zeros((y.shape[0], 2),dtype='float32')
            for i in xrange(y.shape[0]):
                one_hot[i,int(y[i])] = 1.
            y = one_hot
        if shuffle:
            self.shuffle_rng = np.random.RandomState([1,2,3])
            indx = range(self.X.shape[0])
            self.shuffle_rng.shuffle(indx)
            X = X[indx]
            y = y[indx]

        #view_converter = dense_design_matrix.DefaultViewConverter((self.num_feats))
        super(AmazonSmall, self).__init__(X = X, y = y)

        if start is not None:
            assert start >= 0
            if stop > self.X.shape[0]:
                raise ValueError('stop='+str(stop)+'>'+'m='+str(self.X.shape[0]))
            assert stop > start
            self.X = self.X[start:stop,:]
            if self.X.shape[0] != stop - start:
                raise ValueError("X.shape[0]: %d. start: %d stop: %d" % (self.X.shape[0], start, stop))
            if len(self.y.shape) > 1:
                self.y = self.y[start:stop,:]
            else:
                self.y = self.y[start:stop]

        if which_set == 'test':
            assert fit_test_preprocessor is None or (fit_preprocessor == fit_test_preprocessor)

        if self.X is not None and preprocessor:
            preprocessor.apply(self, fit_preprocessor)

    def get_test_set(self):
        args = {}
        args.update(self.args)
        del args['self']
        args['which_set'] = 'test'
        args['start'] = None
        args['stop'] = None
        args['fit_preprocessor'] = args['fit_test_preprocessor']
        args['fit_test_preprocessor'] = None
        return AmazonSmall(**args)

if __name__ == "__main__":
    for cat in AmazonSmall.valid_categories:
        ds = AmazonSmall(which_set = 'train', category=cat, num_feats=784)
        print cat, ds.X.shape
