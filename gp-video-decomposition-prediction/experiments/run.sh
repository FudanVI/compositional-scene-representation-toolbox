#!/bin/bash
export CUDA_VISIBLE_DEVICES='7'
run_file='../src/main.py'
folder_data='/home/ctl/conference/gcm/dataset'
name='clevr'
mode='simple'
num_view=10
path_config='../experiments/'$name'_'$mode'/config_first_stage.yaml'
path_data=$folder_data'/'$name'_multi/'$name'_multi_'$mode'_'$num_view'.h5'
folder_log='../../logs/gp_two_stage_transformer/'$name'_multi_'$mode'_'$num_view'/'random_length_continuous
folder_out='../../outs/gp_two_stage_transformer/'$name'_multi_'$mode'_'$num_view'/'random_length_continuous
python $run_file \
        --path_config $path_config \
        --path_data $path_data \
        --folder_log $folder_log \
        --folder_out $folder_out \
        --train