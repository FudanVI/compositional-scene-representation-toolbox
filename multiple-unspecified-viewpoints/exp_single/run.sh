#!/bin/bash

function basic {
    path_config='config_'$name'.yaml'
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

run_file='../src/main.py'
folder_data='../data'
export CUDA_VISIBLE_DEVICES='0'
for name in 'dsprites' 'abstract'; do
    basic
done
export CUDA_VISIBLE_DEVICES='0,1'
name='clevr'
basic
