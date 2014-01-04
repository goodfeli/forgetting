import numpy as np

from pylearn2.utils import serial

num_jobs = 25

rng = np.random.RandomState([2013, 11, 22])

task_0_template = open('task_0_template.yaml', 'r').read()
task_1_template = open('task_1_template.yaml', 'r').read()

for job_id in xrange(num_jobs):

    h0_col_norm = rng.uniform(1., 5.)
    h1_col_norm = rng.uniform(1., 5.)
    y_col_norm = rng.uniform(1., 5.)

    h0_dim = rng.randint(250, 5000)
    num_pieces_0 = rng.randint(2, 6)
    num_units_0 = h0_dim // num_pieces_0
    h1_dim = rng.randint(250, 5000)
    num_pieces_1 = rng.randint(2, 6)
    num_units_1 = h1_dim // num_pieces_1

    def random_init_string():
        irange = 10. ** rng.uniform(-2.3, -1.)
        return "irange: " + str(irange)

    h0_init = random_init_string()
    h1_init = random_init_string()

    if rng.randint(2):
        y_init = "sparse_init: 0"
    else:
        y_init = random_init_string()

    h0_bias = 0.
    h1_bias = 1.


    learning_rate =  10. ** rng.uniform(-2., -.5)

    if rng.randint(2):
        msat = 2
    else:
        msat = rng.randint(2, 1000)

    final_momentum = rng.uniform(.5, .9)

    lr_sat = rng.randint(200, 1000)

    decay = 10. ** rng.uniform(-3, -1)


    #task_0_yaml_str = task_0_template % locals()

    #serial.mkdir('exp/' + str(job_id))
    #train_file_full_stem = 'exp/'+str(job_id)+'/'
    train_file_full_stem = '{}exp/{}/'.format('/scratch/mmirza/results/forgetting/random_search_sgd_relu_mnist_amazon/', job_id)
    #f = open(train_file_full_stem + 'task_0.yaml', 'w')
    #f.write(task_0_yaml_str)
    #f.close()

    task_1_yaml_str = task_1_template % locals()

    serial.mkdir('exp/' + str(job_id))
    f = open(train_file_full_stem + 'task_1.yaml', 'w')
    f.write(task_1_yaml_str)
    f.close()
