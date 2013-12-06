__author__ = "Xia Da, Ian Goodfellow"
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import tensor as T
from pylearn2.models.mlp import Linear

def lwta(p, block_size):
    """
    The hard local winner take all non-linearity from "Compete to Compute"
    by Rupesh Srivastava et al
    Our implementation differs slightly from theirs--we break ties randomly,
    they break them by earliest index. This difference is just due to ease
    of implementation in theano.
    """
    batch_size = p.shape[0]
    num_filters = p.shape[1]
    num_blocks = num_filters // block_size
    w = p.reshape((batch_size, num_blocks, block_size))
    block_max = w.max(axis=2).dimshuffle(0, 1, 'x') * T.ones_like(w)
    max_mask = T.cast(w >= block_max, 'float32')
    theano_rng = MRG_RandomStreams(20131206 % (2 ** 16))
    denom = max_mask.sum(axis=2).dimshuffle(0, 1, 'x')
    probs = max_mask / denom
    probs = probs.reshape((batch_size * num_blocks, block_size))
    max_mask = theano_rng.multinomial(pvals=probs, dtype='float32')
    max_mask = max_mask.reshape((batch_size, num_blocks, block_size))
    w = w * max_mask
    w = w.reshape((p.shape[0], p.shape[1]))
    return w

class LWTA(Linear):
    """
    An MLP Layer using the LWTA non-linearity.
    """
    def __init__(self, block_size, **kwargs):
        super(LWTA, self).__init__(**kwargs)
        self.block_size = block_size

    def fprop(self, state_below):
        p = super(LWTA, self).fprop(state_below)
        w = lwta(p, self.block_size)
        w.name = self.layer_name + '_out'
        return w
