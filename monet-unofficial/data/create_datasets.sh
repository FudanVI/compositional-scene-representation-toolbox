#!/bin/bash

export CUDA_VISIBLE_DEVICES=''

git submodule update --init

folder_in='multi-object-datasets'
folder_out='.'

if [ ! -d $folder_in ]; then
    gsutil cp -r 'gs://'$folder_in '.'
fi

for name in 'room' 'dsprites' 'clevr'; do
    python 'src/convert_'$name'.py' \
        --folder_in $folder_in \
        --folder_out $folder_out
done
