__author__ = "Ian Goodfellow"

import gc
from matplotlib import pyplot as plt
import numpy as np
import os
import sys

from pylearn2.utils import serial

plt.hold(True)
ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')

i = 0

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#555555']
markers = ['d', 'x', '+', 'o', '<', 'v', '^', 's']


data = serial.load(sys.argv[1])


for old,  new, main_dir in zip(data['old'], data['new'], data['name']):
    plt.plot(old, new, label=main_dir, color=colors[i], marker=markers[i])


    i += 1

plt.legend()
plt.show()
