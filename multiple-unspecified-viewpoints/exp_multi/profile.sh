#!/bin/bash

export CUDA_VISIBLE_DEVICES='0'
num_gpus=$(( (${#CUDA_VISIBLE_DEVICES} + 1) / 2 ))

run_path=../src/main.py
nsys profile \
    --backtrace none \
    --sample none \
    --cpuctxsw none \
    --force-overwrite true \
    --trace cuda,nvtx \
    --output outputs_debug/nsys_profile \
    torchrun \
        --rdzv-backend=c10d \
        --rdzv-endpoint=localhost:0 \
        --nnodes=1 \
        --nproc-per-node=${num_gpus} \
        ${run_path} \
        profiling=true
