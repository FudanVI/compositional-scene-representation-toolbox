#!/bin/bash

folder_src='src'
folder_downloads='downloads'
if [ ! -d $folder_downloads ]; then mkdir $folder_downloads; fi

name_base='shapes'
path_config='config_'$name_base'.yaml'
folder_out='./gray'
python $folder_src'/create_'$name_base'.py' \
    --name $name_base \
    --path_config $path_config \
    --folder_out $folder_out

name_base='mnist'
path_config='config_'$name_base'.yaml'
folder_out='./gray'
python $folder_src'/create_'$name_base'.py' \
    --name $name_base \
    --path_config $path_config \
    --folder_out $folder_out \
    --folder_downloads $folder_downloads

path_npz=$folder_downloads'/dsprites_ndarray.npz'
url='https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
if [ ! -f $path_npz ]; then wget --no-check-certificate $url -O $path_npz; fi
name_base='dsprites'
path_config='config_'$name_base'.yaml'
path_in=$path_npz
folder_out='./gray'
python $folder_src'/create_'$name_base'.py' \
    --name $name_base \
    --path_config $path_config \
    --path_in $path_in \
    --folder_out $folder_out

for name_base in 'shapes' 'mnist' 'dsprites'; do
    path_in='./gray/'$name_base'.h5'
    folder_out='./rgb'
    mode_obj=2
    mode_bck=2
    python $folder_src'/gray_to_rgb.py' \
        --name $name_base \
        --path_in $path_in \
        --folder_out $folder_out \
        --mode_obj $mode_obj \
        --mode_bck $mode_bck
done

path_zip=$folder_downloads'/AbstractScenes_v1.1.zip'
path_extract=$folder_downloads'/AbstractScenes'
url='https://download.microsoft.com/download/4/5/D/45D1BBFC-7944-4AC5-AED2-1A35D85662D7/AbstractScenes_v1.1.zip'
if [ ! -f $path_zip ]; then wget --no-check-certificate $url -O $path_zip; fi
if [ ! -d $path_extract ]; then unzip -qq $path_zip -d $path_extract; fi
name='abstract'
path_config='config_'$name'.yaml'
folder_in=$path_extract'/AbstractScenes_v1.1/Pngs'
folder_out='./rgb'
python $folder_src'/create_'$name'.py' \
    --name $name \
    --path_config $path_config \
    --folder_in $folder_in \
    --folder_out $folder_out

name='clevr'
path_config='config_'$name'.yaml'
folder_base='./clevr-dataset-gen/output/images'
folder_train=$folder_base'/3_6'
folder_general=$folder_base'/7_10'
folder_out='./rgb'
python $folder_src'/create_complex.py' \
    --name $name \
    --path_config $path_config \
    --folder_train $folder_train \
    --folder_general $folder_general \
    --folder_out $folder_out

name='shop'
path_config='config_'$name'.yaml'
folder_base='./shop-vrb-gen/output_single/images'
folder_train=$folder_base'/3_6'
folder_general=$folder_base'/7_10'
folder_out='./rgb'
python $folder_src'/create_complex.py' \
    --name $name \
    --path_config $path_config \
    --folder_train $folder_train \
    --folder_general $folder_general \
    --folder_out $folder_out

for name in 'shapes' 'mnist' 'dsprites' 'abstract' 'clevr' 'shop'; do
    path_in='./rgb/'$name'.h5'
    folder_out='.'
    python $folder_src'/convert.py' \
        --name $name \
        --path_in $path_in \
        --folder_out $folder_out
done
