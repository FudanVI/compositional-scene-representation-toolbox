#!/bin/bash

folder_src='src'
folder_downloads='downloads'
if [ ! -d $folder_downloads ]; then mkdir $folder_downloads; fi

path_npz=$folder_downloads'/dsprites_ndarray.npz'
url='https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
if [ ! -f $path_npz ]; then wget --no-check-certificate $url -O $path_npz; fi
name='dsprites'
name_base=$name'_base'
path_config='config_'$name'.yaml'
path_in=$path_npz
folder_out='.'
python $folder_src'/create_'$name'.py' \
    --name $name_base \
    --path_config $path_config \
    --path_in $path_in \
    --folder_out $folder_out
path_in=$folder_out'/'$name_base'.h5'
python $folder_src'/gray_to_rgb.py' \
    --name $name \
    --path_in $path_in \
    --folder_out $folder_out \
    --mode_obj 2 \
    --mode_bck 2

path_zip=$folder_downloads'/AbstractScenes_v1.1.zip'
path_extract=$folder_downloads'/AbstractScenes'
url='https://download.microsoft.com/download/4/5/D/45D1BBFC-7944-4AC5-AED2-1A35D85662D7/AbstractScenes_v1.1.zip'
if [ ! -f $path_zip ]; then wget --no-check-certificate $url -O $path_zip; fi
if [ ! -d $path_extract ]; then unzip -qq $path_zip -d $path_extract; fi
name='abstract'
path_config='config_'$name'.yaml'
folder_in=$path_extract'/AbstractScenes_v1.1/Pngs'
folder_out='.'
python $folder_src'/create_'$name'.py' \
    --name $name \
    --path_config $path_config \
    --folder_in $folder_in \
    --folder_out $folder_out

for name in 'dsprites' 'abstract'; do
    path_in='./'$name'.h5'
    folder_out='.'
    python $folder_src'/convert.py' \
        --name $name \
        --path_in $path_in \
        --folder_out $folder_out
done

name='clevr'
path_config='config_'$name'.yaml'
folder_base='../clevr-dataset-gen/output/images'
folder_train=$folder_base'/3_6'
folder_general=$folder_base'/7_10'
folder_out='.'
python $folder_src'/create_complex.py' \
    --name $name \
    --path_config $path_config \
    --folder_train $folder_train \
    --folder_general $folder_general \
    --folder_out $folder_out

name='shop'
path_config='config_'$name'.yaml'
folder_base='../shop-vrb-gen/output_single/images'
folder_train=$folder_base'/3_6'
folder_general=$folder_base'/7_10'
folder_out='.'
python $folder_src'/create_complex.py' \
    --name $name \
    --path_config $path_config \
    --folder_train $folder_train \
    --folder_general $folder_general \
    --folder_out $folder_out

name='clevr'
path_config='config_'$name'_multi.yaml'
for idx in {1..2}; do
    for sub_idx in {1..2}; do
        folder_base='../clevr-dataset-gen/output_viewpoint_'$idx'/images_'$sub_idx
        folder_train=$folder_base'/3_6'
        folder_general=$folder_base'/7_10'
        folder_out='.'
        python $folder_src'/create_complex.py' \
            --name $name'_multi_'$idx'_'$sub_idx \
            --path_config $path_config \
            --folder_train $folder_train \
            --folder_general $folder_general \
            --folder_out $folder_out \
            --multiview
        python $folder_src'/create_viewpoint.py' \
            --name $name'_multi_viewpoint_'$idx'_'$sub_idx \
            --path_config $path_config \
            --folder_train $folder_train \
            --folder_general $folder_general \
            --folder_out $folder_out
    done
done

name='shop'
path_config='config_'$name'_multi.yaml'
for idx in {1..2}; do
    for sub_idx in {1..2}; do
        folder_base='../shop-vrb-gen/output_multi_'$idx'/images_'$sub_idx
        folder_train=$folder_base'/3_6'
        folder_general=$folder_base'/7_10'
        folder_out='.'
        python $folder_src'/create_complex.py' \
            --name $name'_multi_'$idx'_'$sub_idx \
            --path_config $path_config \
            --folder_train $folder_train \
            --folder_general $folder_general \
            --folder_out $folder_out \
            --multiview
        python $folder_src'/create_viewpoint.py' \
            --name $name'_multi_viewpoint_'$idx'_'$sub_idx \
            --path_config $path_config \
            --folder_train $folder_train \
            --folder_general $folder_general \
            --folder_out $folder_out
    done
done
