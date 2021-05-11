#!/bin/bash

folder_downloads='downloads'
folder_src='src'
if [ ! -d $folder_downloads ]; then mkdir $folder_downloads; fi

# Download the Static Shapes Dataset used in "Neural Expectation Maximization"
# The Dropbox url is described in https://github.com/sjoerdvansteenkiste/Neural-EM/blob/master/README.md
url_shapes='https://www.dropbox.com/sh/1ue3lrfvbhhkt6s/AABZBL6D1KrCF8CPe-an5psoa/shapes.h5?dl=1'
file_shapes='shapes_28x28_3.h5'
if [ ! -f $folder_downloads/$file_shapes ]; then
    wget $url_shapes -O $folder_downloads/$file_shapes
fi

python $folder_src/create_shapes.py \
    --name 'shapes_20x20' \
    --image_height 20 \
    --image_width 20 \
    --num_objects 3
for num_objects in 2 4; do
    python $folder_src/create_shapes.py \
        --name 'shapes_28x28_'$num_objects \
        --image_height 28 \
        --image_width 28 \
        --num_objects $num_objects
done
python $folder_src/convert.py --folder_downloads $folder_downloads
python $folder_src/combine_shapes.py
python $folder_src/create_mnist.py --folder_downloads $folder_downloads
