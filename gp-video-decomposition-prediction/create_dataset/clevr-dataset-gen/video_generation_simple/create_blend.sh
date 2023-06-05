#!/bin/bash

function func_basic {
    path_run=$1
    folder_out=$2
    batch_size=$3
    num_data=$4
    min_objects=$5
    max_objects=$6
    output_image_dir=$folder_out'/images/'$min_objects'_'$max_objects
    output_scene_dir=$folder_out'/scenes/'$min_objects'_'$max_objects
    output_blend_dir=$folder_out'/blendfiles/'$min_objects'_'$max_objects
    for (( start_idx=0; start_idx<num_data; start_idx=start_idx+batch_size )); do
        blender --background -noaudio --python $path_run -- \
            --save_blendfiles 1 \
            --seed $start_idx \
            --start_idx $start_idx \
            --num_images $batch_size \
            --min_objects $min_objects \
            --max_objects $max_objects \
            --output_image_dir $output_image_dir \
            --output_scene_dir $output_scene_dir \
            --output_blend_dir $output_blend_dir \
            --width 108 \
            --height 80
    done
}

export PYTHONHASHSEED=0

path_run='render_images.py'
folder_out='../output_viewpoint_simple'
batch_size=5

num_data=10200
min_objects=3
max_objects=6
func_basic $path_run $folder_out $batch_size $num_data $min_objects $max_objects

num_data=100
min_objects=7
max_objects=10
func_basic $path_run $folder_out $batch_size $num_data $min_objects $max_objects
