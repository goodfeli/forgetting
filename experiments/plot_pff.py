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

new_, old_, name_ = dumb_sort(data)
i = 0
for old,  new, main_dir in zip(old_, new_, name_):
    plt.plot(old, new, label=main_dir, color=colors[i], marker=markers[i])
    i+=1


# mnist
#plt.legend(bbox_to_anchor=(0.8,1.))
# mnist_amazon
#plt.legend(bbox_to_anchor=(0.87,1.))
#plt.xlim(0.5e-3,1.)
# amazon
plt.legend(bbox_to_anchor=(0.4,1.))


plt.show()
