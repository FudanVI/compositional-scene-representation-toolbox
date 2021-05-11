#!/bin/bash

function convert {
    path=$folder_downloads'/'$name'.h5'
    if [ ! -f $path ]; then wget $url -O $path; fi
    python $file_run \
        --name $name \
        --folder_in $folder_downloads \
        --folder_out $folder_out
}

file_run='src/convert.py'
file_combine='src/combine.py'
folder_downloads='downloads'
folder_out='.'

if [ ! -d $folder_downloads ]; then mkdir $folder_downloads; fi

url='https://www.dropbox.com/sh/1ue3lrfvbhhkt6s/AABZBL6D1KrCF8CPe-an5psoa/shapes.h5?dl=1'
name='shapes'
convert

url='https://www.dropbox.com/sh/1ue3lrfvbhhkt6s/AACy1qUgG82Bj3p0wzh97qlia/flying_shapes.h5?dl=1'
name='flying_shapes'
convert

url='https://www.dropbox.com/sh/1ue3lrfvbhhkt6s/AABoVHgDwMexTQi4kREx9MDBa/flying_shapes_5.h5?dl=1'
name='flying_shapes_5'
convert

url='https://www.dropbox.com/sh/1ue3lrfvbhhkt6s/AACbWwaBZk0sZKKeZrJ_Nv35a/flying_mnist_hard_2digits.h5?dl=1'
name='flying_mnist_hard_2digits'
convert

url='https://www.dropbox.com/sh/1ue3lrfvbhhkt6s/AAAAG7t8HDPwfXDc2R5cYz8Ua/flying_mnist_hard_3digits.h5?dl=1'
name='flying_mnist_hard_3digits'
convert

python $file_combine \
    --name_out 'flying_shapes_3_5' \
    --name_in_1 'flying_shapes' \
    --name_in_2 'flying_shapes_5'
python $file_combine \
    --name_out 'flying_shapes_5_3' \
    --name_in_1 'flying_shapes_5' \
    --name_in_2 'flying_shapes'
python $file_combine \
    --name_out 'flying_mnist_2_3' \
    --name_in_1 'flying_mnist_hard_2digits' \
    --name_in_2 'flying_mnist_hard_3digits'
