import numpy as np

from pylearn2.utils import serial
EXP_PATH = "/RQexec/mirzameh/results/forgetting/random_search_dropout_relu_amazon/"

num_jobs = 25

rng = np.random.RandomState([2013, 11, 22])

task_0_template = open('task_0_template.yaml', 'r').read()
task_1_template = open('task_1_template.yaml', 'r').read()

for job_id in xrange(num_jobs):

    h0_col_norm = rng.uniform(1., 5.)
    h1_col_norm = rng.uniform(1., 5.)
    y_col_norm = rng.uniform(1., 5.)

    h0_dim = rng.randint(250, 5000)
    h1_dim = rng.randint(250, 5000)

    def random_init_string():
        if rng.randint(2):
            sparse_init = rng.randint(10, 30)
            return "sparse_init: " + str(sparse_init)
        irange = 10. ** rng.uniform(-2.3, -1.)
        return "irange: " + str(irange)

    h0_init = random_init_string()
    h1_init = random_init_string()

    if rng.randint(2):
        y_init = "sparse_init: 0"
    else:
        y_init = random_init_string()

    def rectifier_bias():
        if rng.randint(2):
            return 0
        return rng.uniform(0, .3)

    h0_bias = rectifier_bias()
    h1_bias = rectifier_bias()


    learning_rate =  10. ** rng.uniform(-2., -.5)

    if rng.randint(2):
        msat = 2
    else:
        msat = rng.randint(2, 1000)

    final_momentum = rng.uniform(.5, .9)

    lr_sat = rng.randint(200, 1000)

    decay = 10. ** rng.uniform(-3, -1)


    task_0_yaml_str = task_0_template % locals()

    serial.mkdir('{}exp/'.format(EXP_PATH) + str(job_id))
    train_file_full_stem = '{}exp/'.format(EXP_PATH)+str(job_id)+'/'
    f = open(train_file_full_stem + 'task_0.yaml', 'w')
    f.write(task_0_yaml_str)
    f.close()

    task_1_yaml_str = task_1_template % locals()

    serial.mkdir('{}exp/'.format(EXP_PATH) + str(job_id))
    f = open(train_file_full_stem + 'task_1.yaml', 'w')
    f.write(task_1_yaml_str)
    f.close()

