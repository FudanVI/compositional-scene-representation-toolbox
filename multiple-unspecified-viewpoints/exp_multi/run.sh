#!/bin/bash

export CUDA_VISIBLE_DEVICES='0'
name='clevr_multi'
run_file='../src/main.py'
folder_data='../data'
path_config='config_'$name'.yaml'
sub_idx=1
for idx in {1..2}; do
    path_data=$folder_data'/'$name'_'$idx'_'$sub_idx'.h5'
    folder_log='logs/'$name'_'$idx'_'$sub_idx
    folder_out='outs/'$name'_'$idx'_'$sub_idx
    python $run_file \
        --path_config $path_config \
        --path_data $path_data \
        --folder_log $folder_log \
        --folder_out $folder_out \
        --train
done
path_config='config_'$name'_complex.yaml'
sub_idx=2
for idx in {1..2}; do
    path_data=$folder_data'/'$name'_'$idx'_'$sub_idx'.h5'
    path_pretrain='outs/'$name'_'$idx'_1/save_2.pth'
    folder_log='logs/'$name'_'$idx'_'$sub_idx
    folder_out='outs/'$name'_'$idx'_'$sub_idx
    python $run_file \
        --path_config $path_config \
        --path_data $path_data \
        --path_pretrain $path_pretrain \
        --folder_log $folder_log \
        --folder_out $folder_out \
        --train
done
