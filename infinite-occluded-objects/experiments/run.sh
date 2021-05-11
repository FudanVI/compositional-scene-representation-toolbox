#!/bin/bash

function func {
    name=$occlude'_'$color'/'$object
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
path_config='config.yaml'

for occlude in 'sep' 'occ'; do
    for color in 'gray' 'rgb_1' 'rgb_2' 'rgb_3' 'rgb_4'; do
        for object in 'shapes' 'mnist'; do
            func
        done
    done
done
