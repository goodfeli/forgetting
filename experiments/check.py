__author__ = "Ian Goodfellow"

import gc
import numpy as np
import os
import sys

from pylearn2.utils import serial

_, d = sys.argv

fs = os.listdir(d)

best = np.inf

for f in fs:
    try:
        model = serial.load(os.path.join(d, f, 'task_1_best.pkl'))
    except Exception:
        print f, ' not to task 1 yet'
        continue
    try:
        finished_model = serial.load(os.path.join(d, f, 'task_1.pkl'))
    except Exception:
        print f, 'task 1 produced no post-validation output'
    if not finished_model.monitor.training_succeeded:
        print f, 'task 1 had a problem'
        continue
    monitor = model.monitor
    channels = monitor.channels
    def read_channel(s):
        return float(channels[s].val_record[-1])
    task_0_model = serial.load(os.path.join(d, f, 'task_0_best.pkl'))
    monitor = task_0_model.monitor
    v = float(monitor.channels['valid_y_misclass'].val_record[-1])
    print 'job#, orig valid, valid both, new test, old test'
    vb, tn, to = map(read_channel, ['valid_both_y_misclass', 'test_y_misclass', 'test_old_y_misclass'])
    if vb < best:
        best = vb
        print '!', [f, v] + [vb, tn, to]
    else:
        print [f, v] + [vb, tn, to]
    gc.collect()
