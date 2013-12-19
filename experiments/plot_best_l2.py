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

    best = np.inf
    best_channels = None

    for f in fs:
        try:
            finished_model = serial.load(os.path.join(d, f, 'task_1.pkl'))
        except Exception:
            print f, 'task 1 produced no post-validation output'
            assert False
        if not finished_model.monitor.training_succeeded:
            print f, 'task 1 had a problem'
            continue
        monitor = finished_model.monitor
        channels = monitor.channels

        old = channels['test_old_y_misclass'].val_record
        new = channels['test_y_misclass'].val_record

        l2 = [np.sqrt(o ** 2. + n **2.) for o, n in zip(old, new)]
        v = min(l2)

        if v < best:
            best = v
            best_channels = finished_model.monitor.channels
        gc.collect()

    old = best_channels['test_old_y_misclass'].val_record
    new = best_channels['test_y_misclass'].val_record

    filtered = [elem for elem in zip(old, new) if max(elem) < .1]

    old = [elem[0] for elem in filtered]
    new = [elem[1] for elem in filtered]

    plt.scatter(old, new, label=main_dir, color=colors[i], marker=markers[i])

    i += 1

plt.legend()
plt.show()
