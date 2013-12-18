__author__ = "Ian Goodfellow"

import gc
from matplotlib import pyplot as plt
import numpy as np
import os
import sys

from pylearn2.utils import serial

plt.hold(True)

i = 0

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#555555']
markers = ['d', 'x', '+', 'o', '<', 'v', '^', 's']

for main_dir in sys.argv[1:]:
    print main_dir
    d = os.path.join(main_dir, 'exp')

    fs = os.listdir(d)

    first = True
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

        old = finished_model.monitor.channels['test_old_y_misclass'].val_record
        new = finished_model.monitor.channels['test_y_misclass'].val_record

        filtered = [elem for elem in zip(old, new) if max(elem) < .1]

        if len(filtered) == 0:
            continue

        old = [elem[0] for elem in filtered]
        new = [elem[1] for elem in filtered]

        if first:
            plt.scatter(old, new, label=main_dir, color=colors[i], marker=markers[i])
        else:
            plt.scatter(old, new, color=colors[i], marker=markers[i])
        first = False

        gc.collect()



    i += 1

plt.legend()
plt.show()
