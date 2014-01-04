__author__ = "Ian Goodfellow"

import gc
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import pickle



plt.hold(True)
ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')


colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#555555']
markers = ['d', 'x', '+', 'o', '<', 'v', '^', 's']

data = pickle.load(open(sys.argv[1], 'r'))

def dumb_sort(data):
    sort_keys = ['sgd_sigmoid', 'dropout_sigmoid', 'sgd_relu', 'dropout_relu', 'sgd_maxout', 'dropout_maxout', 'sgd_lwta', 'dropout_lwta']
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
    def captalize(name):
        if name == 'dropout':
            return 'Dropout'
        if name == 'sgd':
            return 'SGD'
        if name == 'maxout':
            return 'Maxout'
        if name == 'relu':
            return 'ReLUs'
        if name == 'lwta':
            return 'LWTA'
        if name == 'sigmoid':
            return 'Sigmoid'

    name = name.split('_')
    return "{}, {}".format(captalize(name[2]), captalize(name[3]))

new_, old_, name_ = dumb_sort(data)
i = 0
for old,  new, main_dir in zip(old_, new_, name_):
    plt.plot(old, new, label= convert_name(main_dir), color=colors[i], marker=markers[i])
    i+=1


# mnist
if sys.argv[2] == 'mnist':
    plt.legend(bbox_to_anchor=(0.8,1.))
    plt.xlabel('Test error, old task')
    plt.ylabel('Test error, new task')
    plt.title('Old task: MNIST, New task: MNIST permutation')

# mnist_amazon
if sys.argv[2] == 'mnist_amazon':
    plt.xlabel('Test error, old task')
    plt.ylabel('Test error, new task')
    plt.title('Old task: MNIST (2,9), New task: Amazon (DVD)')
    plt.legend(bbox_to_anchor=(0.87,1.))
    plt.xlim(0.5e-3,1.)

# amazon
if sys.argv[2] == 'amazon':
    plt.legend(bbox_to_anchor=(0.4,1.))
    plt.xlabel('Test error, old task')
    plt.ylabel('Test error, new task')
    plt.title('Old task: Amazon (Kitchen), New task: Amazon (DVD)')


plt.show()
