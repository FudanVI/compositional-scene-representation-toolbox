#!/bin/bash

export CUDA_VISIBLE_DEVICES='0'
num_gpus=$(( (${#CUDA_VISIBLE_DEVICES} + 1) / 2 ))

run_path=../src/main.py
torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=1 \
    --nproc-per-node=${num_gpus} \
    ${run_path}
