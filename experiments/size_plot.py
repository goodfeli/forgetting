import sys
import os
import gc
import numpy as np
from pylearn2.utils import serial
from pylearn2.scripts.num_parameters import num_parameters
import matplotlib.pyplot as plt
import ipdb


def extract_data(task_0, task_1):
    model = serial.load(task_0)
    num_params = num_parameters(model)

    valid_0 = model.monitor.channels['valid_y_misclass'].val_record[-1]

    model = serial.load(task_1)
    valid_1 = model.monitor.channels['valid_both_y_misclass'].val_record[-1]

    return num_params, float(valid_0), float(valid_1)


def collect_all(main_path, dataset):
    all_data = []
    for method in ['sgd', 'dropout']:
        for act in ['maxout', 'relu', 'lwta', 'sigmoid']:
            print method, act
            path = os.path.join(main_path, "random_search_{}_{}_{}".format(method, act, dataset), "exp")
            dirs = os.listdir(path)
            num_params = []
            val_0 = []
            val_1 = []
            for d in dirs:
                p, v0, v1 = extract_data(os.path.join(path, d, 'task_0_best.pkl'), os.path.join(path, d, 'task_1_best.pkl'))
                num_params.append(p)
                val_0.append(v0)
                val_1.append(v1)
                gc.collect()
            data = {}
            data['method'] = method
            data['act'] = act
            data['num_params'] = num_params
            data['val_0'] = val_0
            data['val_1'] = val_1
            all_data.append(data)

    return all_data


def plot(data, ds):

    colors = ['b', 'g', 'r', 'k', 'm', 'y', 'k', '#555555']
    act_names = ['maxout', 'relu', 'lwta', 'sigmoid']
    for i, item in enumerate(data):
        if item['method'] == 'sgd':
            marker = 'o'
        else:
            marker = 'd'
        color = colors[act_names.index(item['act'])]
        min = np.argmin(np.asarray(item['val_1']))
        plt.scatter(item['num_params'][min], item['val_1'][min],
                label = "{}_{}".format(item['method'], item['act']),
                color = color, marker = marker, s = 100)

    plt.legend()
    ax = plt.gca()
    ax.yaxis.grid(True, 'minor')
    ax.yaxis.grid(True, 'major')
    plt.xlabel('# of parameters')
    plt.ylabel('Sum of validation errors of both tasks')
    plt.yscale('log')
    if ds == 'mnist':
        #plt.ylim(0.01, 0.1)
        #plt.xlim(1e06,2e7)
        plt.title("Old task: MNIST, New task: MNIST permutation")
    elif ds == 'amazon':
        plt.title("Old task: Amazon (Kitchen), New task: Amazon (DVD)")
    else:
        plt.title("Old task: MNIST (2,9), New task: Amazon (DVD)")
    plt.show()


def print_results(data, ds):
    print "{}: ".format(ds)
    for item in data:
        min = np.argmin(np.asarray(item['val_1']))
        #print min
        num_param = item['num_params'][min]
        val = np.asarray(item['val_0'][min]) + np.asarray(item['val_1'][min])
        print "{}_{}: # param: {}, val: {}".format(item['method'], item['act'], num_param, val)


def whisker(data, ds):

    vec = [item['val_0'] for item in data]
    label = ["{}_{}".format(item['method'], item['act']) for item in data]
    plt.boxplot(vec)
    plt.xticks(range(1, len(label) +1), label)
    plt.yscale('log')
    plt.ylabel("Validation error")
    if ds == 'mnist':
        plt.title("MNIST")
    elif ds == 'amazon':
        plt.title("Amazon (Kitchen)")
    else:
        plt.title("MNIST (2,9)")

    plt.show()

if __name__ == "__main__":

    _, path, ds, extract = sys.argv

    if int(extract) == 1:
        data = collect_all(path, ds)
        serial.save("{}_val_data.pkl".format(ds), data)
    else:
        data = serial.load(os.path.join(path, "{}_val_data.pkl".format(ds)))

    plot(data, ds)
    #print_results(data, ds)
    #whisker(data, ds)
