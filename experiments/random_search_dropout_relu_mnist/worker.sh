#!/bin/bash
cd $1
train.py task_0.yaml || exit -1
echo "starting task 1"
train.py task_1.yaml
