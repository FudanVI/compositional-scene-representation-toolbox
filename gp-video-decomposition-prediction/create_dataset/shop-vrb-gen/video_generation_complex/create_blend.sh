#!/bin/bash

function func_basic {
    batch_size=$1
    num_data=$2
    min_objects=$3
    max_objects=$4
    output_image_dir=$folder_out'/images/'$min_objects'_'$max_objects
    output_scene_dir=$folder_out'/scenes/'$min_objects'_'$max_objects
    output_blend_dir=$folder_out'/blendfiles/'$min_objects'_'$max_objects
    for (( start_idx=0; start_idx<num_data; start_idx=start_idx+batch_size )); do
        blender --background -noaudio --python $path_run -- \
            --seed $start_idx \
            --start_idx $start_idx \
            --num_images $batch_size \
            --min_objects $min_objects \
            --max_objects $max_objects \
            --output_image_dir $output_image_dir \
            --output_scene_dir $output_scene_dir \
            --output_blend_dir $output_blend_dir \
            --width 108 \
            --height 80 \
            --save_blendfiles
    done
}

export PYTHONHASHSEED=0

path_run='render_images.py'
folder_out='../output_multi_1_complex'

batch_size=1
num_data=10200
min_objects=3
max_objects=6
func_basic $batch_size $num_data $min_objects $max_objects

batch_size=1
num_data=100
min_objects=7
max_objects=10
func_basic $batch_size $num_data $min_objects $max_objects
