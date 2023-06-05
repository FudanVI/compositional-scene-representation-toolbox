#!/bin/bash

path_run='rename_files.py'
folder_base='output_multi_1'
folder_scene='../'$folder_base'/scenes'
folder_blend='../'$folder_base'/blendfiles'
blender --background -noaudio --python $path_run -- \
    --folder_scene $folder_scene \
    --folder_blend $folder_blend
