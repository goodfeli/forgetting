#!/bin/bash
cd $1
python ~/projects/pylearn2/pylearn2/scripts/train.py task_0.yaml || exit -1
echo "starting task 1"
python ~/projects/pylearn2/pylearn2/scripts/train.py task_1.yaml
