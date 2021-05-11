#!/bin/bash

export CUDA_VISIBLE_DEVICES='0'
run_file='../src/main.py'
folder_data='../data'

mode='single'
coef_mask='1.0'
coef_kld='10.0'
for name in 'mnist' 'abstract'; do
    path_config='config_'$name'.yaml'
    path_data=$folder_data'/'$name'.h5'
    name_dir=$name'_'$coef_mask'_'$coef_kld
    folder_log='logs_'$mode'/'$name_dir
    folder_out='outs_'$mode'/'$name_dir
    python $run_file \
        --path_config $path_config \
        --path_data $path_data \
        --folder_log $folder_log \
        --folder_out $folder_out \
        --coef_kld $coef_kld \
        --coef_mask $coef_mask \
        --mode_single \
        --train
done

mode='direct'
coef_mask='0.0'
coef_kld='1.0'
for name in 'mnist' 'abstract'; do
    path_config='config_'$name'.yaml'
    path_data=$folder_data'/'$name'.h5'
    for split_data in {0..1}; do
        name_dir=$name'_'$split_data'_'$coef_kld
        folder_log='logs_'$mode'/'$name_dir
        folder_out='outs_'$mode'/'$name_dir
        python $run_file \
            --path_config $path_config \
            --path_data $path_data \
            --folder_log $folder_log \
            --folder_out $folder_out \
            --split_train $split_data \
            --split_test $split_data \
            --coef_kld $coef_kld \
            --coef_mask $coef_mask \
            --train
    done
done

mode='adi'
coef_mask='1.0'
coef_kld='1.0'
for name in 'mnist' 'abstract'; do
    path_config='config_'$name'.yaml'
    path_data=$folder_data'/'$name'.h5'
    path_pretrain='outs_single/'$name'_1.0_10.0/model.pickle'
    for split_data in {0..1}; do
        name_dir=$name'_'$split_data'_'$coef_kld'_'$coef_mask
        folder_log='logs_'$mode'/'$name_dir
        folder_out='outs_'$mode'/'$name_dir
        python $run_file \
            --path_config $path_config \
            --path_data $path_data \
            --path_pretrain $path_pretrain \
            --folder_log $folder_log \
            --folder_out $folder_out \
            --split_train $split_data \
            --split_test $split_data \
            --coef_kld $coef_kld \
            --coef_mask $coef_mask \
            --train
    done
done
