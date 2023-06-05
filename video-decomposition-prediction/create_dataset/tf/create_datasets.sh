#!/bin/bash

export CUDA_VISIBLE_DEVICES=''

folder_src='src'
folder_in='..'
folder_out='.'

#for name in 'shapes' 'mnist' 'dsprites' 'abstract' 'clevr' 'shop'; do
#    python $folder_src'/convert_tfrecord.py' \
#        --folder_in $folder_in \
#        --folder_out $folder_out \
#        --name $name
#done


name='shop'
python $folder_src'/convert_tfrecord.py' \
    --folder_in $folder_in'/'$name \
    --folder_out $folder_out'/'$name \
    --name $name