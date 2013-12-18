__author__ = "Ian Goodfellow"

import gc
import numpy as np
import os
import sys
import pickle

from pylearn2.utils import serial

_, d, name = sys.argv

fs = os.listdir(d)

best = np.inf

results = {'name' : [], 'test_old' : [], 'test_new' : []}
for f in fs:
    try:
        model = serial.load(os.path.join(d, f, 'task_1_best.pkl'))
    except Exception:
        print f, ' not to task 1 yet'
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
    results['name'].append(os.path.join(d, f, 'task_1_best.pkl'))
    results['test_old'].append(to)
    results['test_new'].append(tn)
    if vb < best:
        best = vb
        print '!', [f, v] + [vb, tn, to]
    else:
        print [f, v] + [vb, tn, to]
    gc.collect()

with open("results/{}.pkl".format(name), 'w') as outf:
    pickle.dump(results, outf)
