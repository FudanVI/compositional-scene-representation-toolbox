#!/bin/bash

function run_model_1 {
    sub_idx=1
    path_config='config_'$name'.yaml'
    path_data=$folder_data'/'$name'_'$idx'_'$sub_idx'.h5'
    folder_log='logs/'$name'_'$idx'_'$sub_idx
    folder_out='outs/'$name'_'$idx'_'$sub_idx
    python $run_file \
        --path_config $path_config \
        --path_data $path_data \
        --folder_log $folder_log \
        --folder_out $folder_out \
        --without_oc \
        --train
}

function run_model_2 {
    sub_idx=2
    path_config='config_'$name'_complex.yaml'
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
        --without_oc \
        --train
}

run_file='../src/main.py'
folder_data='../data'

for name in 'clevr_multi' 'shop_multi'; do
    for idx in 1 2; do
        run_model_1
        run_model_2
    done
done
