#!/bin/bash

function func {
    folder_image='../'$folder_base'/images_'$idx_data
    folder_scene='../'$folder_base'/scenes'
    folder_blend='../'$folder_base'/blendfiles'
    blender --background -noaudio --python $path_run -- \
        --folder_image $folder_image'/'$num_objects \
        --folder_scene $folder_scene'/'$num_objects \
        --folder_blend $folder_blend'/'$num_objects \
        --offset $offset \
        --num_images $num_images \
        --num_views $num_views \
        --min_theta $min_theta \
        --max_theta $max_theta \
        --min_phi $min_phi \
        --max_phi $max_phi \
        --min_rho $min_rho \
        --max_rho $max_rho
}

export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES='5'

path_run='blend_to_pngs.py'
folder_base='output_multi_1'

idx_data=1
num_views=10
min_theta='0.0'
max_theta='0.5'
min_phi='0.3'
max_phi='0.5'
min_rho='10.75'
max_rho='11.75'
num_objects='3_6'
offset=0
num_images=5200
func
num_objects='7_10'
offset=0
num_images=100
func

idx_data=2
num_views=20
min_theta='0.0'
max_theta='1.0'
min_phi='0.3'
max_phi='0.6'
min_rho='10.5'
max_rho='12.0'
num_objects='3_6'
offset=0
num_images=5200
func
num_objects='7_10'
offset=0
num_images=100
func
