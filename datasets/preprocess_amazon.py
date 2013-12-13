import numpy as np
import ipdb

PATH = "/data/lisa/data/multi_domain_sentiment_analysis/acl_07/pylibsvm-data/"
SAVE_PATH = "/data/lisa/data/multi_domain_sentiment_analysis/acl_07/numpy-data/"
NUM_FEAT = 5000

def preporcess(file_path, num_feat):
    """
    Reads Xavier's preprocessed data into binary fromat and
    save as npy file
    """

    def dict_vec(data, num_feat):
        rval = np.zeros(num_feat)
        for i in xrange(num_feat):
            if data.has_key(i):
                rval[i] = data[i]

        return rval

    f = open(file_path, 'r')
    feats = []
    for line in f:
        line = line.rstrip(' \n').split(' ')
        dic_feat = ({int(key) - 1: int(value) for (key, value) in \
                [item.split(':') for item in line]})
        feats.append(dict_vec(dic_feat, num_feat))
    return np.concatenate([item[np.newaxis, :] for item in feats])

def label(file_path):
    ipdb.set_trace()

if __name__ == "__main__":

    for set in ['train', 'test']:
        for cat in ['kitchen', 'dvd', 'books', 'electronics']:
            path = "{}in-domain-{}-{}-{}.vec".format(PATH,set, cat, NUM_FEAT)
            x = preporcess(path, NUM_FEAT)
            path = "{}in-domain-{}-{}-{}.lab".format(PATH,set, cat, NUM_FEAT)
            y = np.loadtxt(path)

            path = "{}in-domain-{}-{}-{}-x.npy".format(SAVE_PATH, set, cat, NUM_FEAT)
            np.save(path, x)
            path = "{}in-domain-{}-{}-{}-y.npy".format(SAVE_PATH, set, cat, NUM_FEAT)
            np.save(path, y)

        #all
        path = "{}all-domain-{}-{}.vec".format(PATH,set, NUM_FEAT)
        x = preporcess(path, NUM_FEAT)
        path = "{}all-domain-{}-{}.lab".format(PATH,set, NUM_FEAT)
        y = np.loadtxt(path)

        path = "{}in-domain-{}-{}-{}-x.npy".format(SAVE_PATH, set, 'all', NUM_FEAT)
        np.save(path, x)
        path = "{}in-domain-{}-{}-{}-y.npy".format(SAVE_PATH, set, 'all', NUM_FEAT)
        np.save(path, y)


