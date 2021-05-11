#!/bin/bash

folder_src='src'
folder_downloads='downloads'
if [ ! -d $folder_downloads ]; then mkdir $folder_downloads; fi

folder_out='.'
name='mnist'
python $folder_src'/create_'$name'.py' \
    --path_config 'config_'$name'.yaml' \
    --folder_downloads $folder_downloads \
    --folder_out $folder_out

url='https://vision.ece.vt.edu/clipart/dataset/AbstractScenes_v1.1.zip'
path_zip=$folder_downloads'/AbstractScenes_v1.1.zip'
path_extract=$folder_downloads'/AbstractScenes_v1.1'
if [ ! -f $path_zip ]; then wget --no-check-certificate $url -O $path_zip; fi
if [ ! -d $path_extract ]; then unzip -qq $path_zip -d $folder_downloads; fi

folder_in=$path_extract'/Pngs'
folder_out='.'
name='abstract'
python $folder_src'/create_'$name'.py' \
    --path_config 'config_'$name'.yaml' \
    --folder_in $folder_in \
    --folder_out $folder_out
