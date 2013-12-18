import numpy as np
import matplotlib.pyplot as plt
from pylearn2.utils import serial


EXPS = ['dropout_lwta_amazon', 'sgd_lwta_amazon']

for item in EXPS:
    data = serial.load("results/{}.pkl".format(item))
    sort_indx = sorted(range(len(data['test_old'])), key=lambda k: data['test_old'][k])
    plt.plot(np.asarray(data['test_old'])[sort_indx],
            np.asarray(data['test_new'])[sort_indx],
            '-o', label = "radndom_search_{}".format(item))
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.show()


