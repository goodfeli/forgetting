"""
This script makes a dataset of amazon 5000 reduced to 784 wit PCA

"""

from pylearn2.utils import serial
from pylearn2.datasets import preprocessing
from pylearn2.utils import string_utils
from forgetting.datasets.amazon import AmazonSmall

data_dir = string_utils.preprocess('${PYLEARN2_DATA_PATH}/multi_domain_sentiment_analysis/acl_07/numpy-data')

preprocessor = preprocessing.PCA(num_components = 784)

for category in AmazonSmall.valid_categories:
    train = AmazonSmall(which_set = 'train', category = category)
    train.apply_preprocessor(preprocessor = preprocessor, can_fit = True)
    train.use_design_loc(data_dir+'/design_loc_train_{}.npy'.format(category))
    serial.save(data_dir+ '/in-domain-train-{}-784-x.npy'.format(category), train)

    test = AmazonSmall(which_set = 'test', category = category)
    test.apply_preprocessor(preprocessor = preprocessor, can_fit = False)
    test.use_design_loc(data_dir+'/design_loc_test_{}.npy'.format(category))
    serial.save(data_dir+ '/in-domain-test-{}-784-x.npy'.format(category), test)

    serial.save(data_dir + '/{}-preprocessor.pkl'.format(category),preprocessor)

