import numpy as np

from pylearn2.utils import serial

num_jobs = 25

rng = np.random.RandomState([2013, 11, 22])

task_0_template = open('task_0_template.yaml', 'r').read()
task_1_template = open('task_1_template.yaml', 'r').read()
task_2_template = open('task_2_template.yaml', 'r').read()

channel_sizes = [16, 32, 64, 96, 128]

for job_id in xrange(num_jobs):

    h0_ker_norm = rng.uniform(1., 5.)
    h1_ker_norm = rng.uniform(1., 5.)
    h2_ker_norm = rng.uniform(1., 5.)
    h3_col_norm = rng.uniform(1., 5.)
    h4_col_norm = rng.uniform(1., 5.)
    y_col_norm = rng.uniform(1., 5.)

    h0_num_chan = channel_sizes[rng.randint(len(channel_sizes))]
    h1_num_chan = channel_sizes[rng.randint(len(channel_sizes))]
    h2_num_chan = channel_sizes[rng.randint(len(channel_sizes))]

    piece_sizes = [1,2,3]
    h0_num_pieces = piece_sizes[rng.randint(len(piece_sizes))]
    h1_num_pieces = piece_sizes[rng.randint(len(piece_sizes))]
    h2_num_pieces = piece_sizes[rng.randint(len(piece_sizes))]

    h3_dim = rng.randint(250, 5000)
    h3_num_pieces = rng.randint(2, 6)
    h3_num_units = h3_dim //h3_num_pieces
    h4_dim = rng.randint(250, 5000)
    h4_num_pieces = rng.randint(2, 6)
    h4_num_units = h4_dim //h4_num_pieces

    def random_init_string():
        irange = 10. ** rng.uniform(-2.3, -1.)
        return "irange: " + str(irange)

    h0_init = random_init_string()
    h1_init = random_init_string()
    h2_init = random_init_string()
    h3_init = random_init_string()
    h4_init = random_init_string()

    if rng.randint(2):
        y_init = "sparse_init: 0"
    else:
        y_init = random_init_string()

    h3_bias = 1.
    h4_bias = 1.


    learning_rate =  10. ** rng.uniform(-2., -.5)

    if rng.randint(2):
        msat = 2
    else:
        msat = rng.randint(2, 1000)

    final_momentum = rng.uniform(.5, .9)

    lr_sat = rng.randint(200, 1000)

    decay = 10. ** rng.uniform(-3, -1)


    #task_0_yaml_str = task_0_template % locals()

    ##serial.mkdir('exp/' + str(job_id))
    train_file_full_stem = 'exp/'+str(job_id)+'/'
    #f = open(train_file_full_stem + 'task_0.yaml', 'w')
    #f.write(task_0_yaml_str)
    #f.close()

    #task_1_yaml_str = task_1_template % locals()

    ##serial.mkdir('exp/' + str(job_id))
    #f = open(train_file_full_stem + 'task_1.yaml', 'w')
    #f.write(task_1_yaml_str)
    #f.close()

    task_2_yaml_str = task_2_template % locals()

    #serial.mkdir('exp/' + str(job_id))
    f = open(train_file_full_stem + 'task_2.yaml', 'w')
    f.write(task_2_yaml_str)
    f.close()
