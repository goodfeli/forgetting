__author__ = "Ian Goodfellow"

import gc
import numpy as np
import os
import sys
from pylearn2.utils import serial
import ipdb



i = 0


def get_val(main_dir):
    d = os.path.join(main_dir, 'exp')
    fs = os.listdir(d)

    x = []
    for f in fs:
        model = serial.load(os.path.join(d, f, 'task_0_best.pkl'))
        x.append(float(model.monitor.channels['test_y_misclass'].val_record[-1]))

        gc.collect()

    return x, fs


def do_stuff(main_path, exp, ds):
    #sgd
    path = os.path.join(main_path, "random_search_sgd_{}_{}/".format(exp, ds))
    x, fs =get_val(path)
    print "sgd", fs[np.argmin(x)]

    path = os.path.join(main_path, "random_search_dropout_{}_{}/".format(exp, ds))
    x, fs =get_val(path)
    print "dropout", fs[np.argmin(x)]



if __name__ == "__main__":
    path = sys.argv[1]
    exp = sys.argv[2]
    ds = sys.argv[3]
    do_stuff(path, exp, ds)
