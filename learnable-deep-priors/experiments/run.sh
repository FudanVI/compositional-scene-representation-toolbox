#!/bin/bash

function func {
    path_data=$folder_data'/'$name'.h5'
    folder_log='logs/'$name
    folder_out='outs/'$name
    python $run_file \
        --path_config $path_config \
        --path_data $path_data \
        --folder_log $folder_log \
        --folder_out $folder_out \
        --train
}

export CUDA_VISIBLE_DEVICES='0'
run_file='../src/main.py'
folder_data='../data'

for name in 'shapes_20x20' 'shapes_28x28'; do
    path_config='config_'$name'.yaml'
    func
done

path_config='config_mnist.yaml'
for name in 'mnist_20' 'mnist_500' 'mnist_all'; do
    func
done
