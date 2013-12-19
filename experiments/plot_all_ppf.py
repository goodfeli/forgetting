__author__ = "Ian Goodfellow"

import gc
from matplotlib import pyplot as plt
import numpy as np
import os
import sys

from pylearn2.utils import serial

from forgetting.ppf import cloud_to_ppf

plt.hold(True)
ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')

i = 0

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#555555']
markers = ['d', 'x', '+', 'o', '<', 'v', '^', 's']

for main_dir in sys.argv[1:]:
    print main_dir
    d = os.path.join(main_dir, 'exp')

    fs = os.listdir(d)

    x = []
    y = []
    for f in fs:
        """
        try:
            model = serial.load(os.path.join(d, f, 'task_1_best.pkl'))
        except Exception:
            print f, ' not to task 1 yet'
            continue
        """
        try:
            finished_model = serial.load(os.path.join(d, f, 'task_1.pkl'))
        except Exception:
            print f, 'task 1 produced no post-validation output'
            assert False
        if not finished_model.monitor.training_succeeded:
            print f, 'task 1 had a problem'
            continue
        """
        monitor = model.monitor
        channels = monitor.channels
        def read_channel(s):
            return float(channels[s].val_record[-1])
        """

        x += finished_model.monitor.channels['test_old_y_misclass'].val_record
        y += finished_model.monitor.channels['test_y_misclass'].val_record

        gc.collect()
    old, new = cloud_to_ppf(x, y, False)
    plt.plot(old, new, label=main_dir, color=colors[i], marker=markers[i])


    i += 1

plt.legend()
plt.show()
