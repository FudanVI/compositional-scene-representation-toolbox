#!/bin/bash

file_run='src/create_mnist.py'
file_combine='src/combine.py'

python $file_run --name 'mnist_0_1_2' --num_objects_list 0 1 2
python $file_run --name 'mnist_0_1_3' --num_objects_list 0 1 3
python $file_run --name 'mnist_2' --num_objects_list 2 --num_train 0
python $file_run --name 'mnist_3' --num_objects_list 3 --num_train 0
python $file_combine \
    --name_out 'mnist_extrapol' \
    --name_in_1 'mnist_0_1_2' \
    --name_in_2 'mnist_3'
python $file_combine \
    --name_out 'mnist_interpol' \
    --name_in_1 'mnist_0_1_3' \
    --name_in_2 'mnist_2'
