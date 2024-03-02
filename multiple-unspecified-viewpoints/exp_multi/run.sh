#!/bin/bash

function run_training {
    run_runs_name=num_runs_${mode}
    num_runs=${!run_runs_name}
    for (( idx_run = 0; idx_run < num_runs; idx_run = idx_run + 1 )); do
        echo ${data_name}_${mode}_run_${idx_run}
        torchrun \
            --rdzv-backend=c10d \
            --rdzv-endpoint=localhost:0 \
            --nnodes=1 \
            --nproc-per-node=${num_gpus} \
            ${run_path} \
            --config-path ${config_path} \
            --config-name ${config_name_prefix}_${mode} \
            data_name=${data_name} \
            hydra.run.dir=${folder_out}/${data_name}_${mode}_run_${idx_run}
        echo
    done
    python ${config_path}/select_best.py \
        --folder ${folder_out} \
        --name ${data_name}_${mode} \
        --num_runs ${num_runs}
}

export CUDA_VISIBLE_DEVICES='0'
num_gpus=$(( (${#CUDA_VISIBLE_DEVICES} + 1) / 2 ))

run_path=../src/main.py
config_path=../exp_multi
folder_out=outputs

num_runs_single=10
num_runs_multi=5

config_name_prefix=config_blender
for data_name in clevr shop; do
    for mode in single multi; do
        run_training
    done
done

config_name_prefix=config_kubric
for data_name in gso shapenet; do
    for mode in single multi; do
        run_training
    done
done
