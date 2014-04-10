__author__ = "Ian Goodfellow"

import gc
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import pickle
import ipdb


plt.hold(True)
ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')


colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#555555']
markers = ['d', 'x', '+', 'o', '<', 'v', '^', 's']

data = pickle.load(open(sys.argv[1], 'r'))
#ipdb.set_trace()


def dumb_sort(data):
    sort_keys = ['sgd_sigmoid', 'dropout_sigmoid', 'sgd_relu', 'dropout_relu', 'sgd_maxout', 'dropout_maxout', 'sgd_lwta', 'dropout_lwta']
    sort_keys = [ 'relu_mnist_025', 'dropout_relu_mnist/', 'relu_mnist_075', 'sgd_relu',]
    new = []
    old = []
    name = []
    for item in sort_keys:
        for i in xrange(len(data['name'])):
                if item in data['name'][i]:
                    new.append(data['new'][i])
                    old.append(data['old'][i])
                    name.append(data['name'][i])
                    continue
    return new, old, name

def convert_name(name):
    if name == '/data/lisatmp/goodfeli/random_search_dropout_relu_mnist_025/':
        #return 'Dropout p=0.25'
        return 'Inclusion p=0.25'
    if name == '/data/lisatmp/goodfeli/random_search_dropout_relu_mnist_075/':
        #return 'Dropout p=0.75'
        return 'Inclusion p=0.75'
    if name == '/u/goodfeli/forgetting/experiments/random_search_dropout_relu_mnist/':
        #return 'Dropout p=0.5'
        return 'Inclusion p=0.5'
    if name == '/u/goodfeli/forgetting/experiments/random_search_sgd_relu_mnist/':
        #return 'Dropout p=1.0'
        return 'Inclusion p=1.0'



new_, old_, name_ = dumb_sort(data)
#new_, old_, name_ = data['new'], data['old'], data['name']
i = 0
for old,  new, main_dir in zip(old_, new_, name_):
    plt.plot(old, new, label= convert_name(main_dir), color=colors[i], marker=markers[i])
    i+=1


# mnist
plt.legend(bbox_to_anchor=(0.8,1.), prop={'size':20})
#plt.legend()
plt.xlabel('Test error, old task')
plt.ylabel('Test error, new task')
plt.title('Old task: MNIST, New task: MNIST permutation, Activation: ReLUs')


plt.show()
