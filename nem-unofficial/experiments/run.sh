#!/bin/bash

function func {
    path_data=$folder_data'/'$file_data'.h5'
    path_config='config_'$name'.yaml'
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

name='shapes'
file_data='shapes'
func

file_data='flying_shapes_3_5'
for name in 'flying_shapes_3_3' 'flying_shapes_3_5'; do
    func
done
file_data='flying_shapes_5_3'
for name in 'flying_shapes_5_3' 'flying_shapes_5_5'; do
    func
done

name='flying_mnist'
file_data='flying_mnist_2_3'
func
