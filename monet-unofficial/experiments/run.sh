#!/bin/bash

function func {
    path_config='config_'$name'.yaml'
    folder_log='logs/'$name
    folder_out='outs/'$name
    python $run_file \
        --path_config $path_config \
        --folder_data $folder_data \
        --folder_log $folder_log \
        --folder_out $folder_out \
        --train
}

run_file='../src/main.py'
folder_data='../data'

for name in 'room' 'dsprites' 'clevr'; do
    func
done
