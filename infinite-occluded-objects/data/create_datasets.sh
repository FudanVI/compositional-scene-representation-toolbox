#!/bin/bash

folder_src='src'

# Gray
python $folder_src'/download.py'
for occlusion in 0 1; do
    if [ $occlusion == 0 ]; then
        folder_outputs='sep_gray'
    else
        folder_outputs='occ_gray'
    fi
    if [ ! -d $folder_outputs ]; then mkdir $folder_outputs; fi
    for mode_object in 'shapes' 'mnist'; do
        for num_objects_all in '2 3' '4'; do
            if [[ $num_objects_all == '2 3' ]]; then
                num_train=50000
                num_valid=10000
                num_test=10000
            else
                num_train=0
                num_valid=0
                num_test=10000
            fi
            name=$mode_object'_'${num_objects_all// /_}
            python $folder_src'/create_'$mode_object'.py' \
                --name $name \
                --folder_outputs $folder_outputs \
                --occlusion $occlusion \
                --num_objects_all $num_objects_all \
                --num_train $num_train \
                --num_valid $num_valid \
                --num_test $num_test
        done
    done
done

# RGB
for only_object in 1 0; do
    for same_color in 0 1; do
        data_id=$(( (1 - only_object) * 2 + same_color + 1 ))
        for mode_occlude in 'sep' 'occ'; do
            folder_inputs=$mode_occlude'_gray'
            folder_outputs=$mode_occlude'_rgb_'$data_id
            if [ ! -d $folder_outputs ]; then mkdir $folder_outputs; fi
            for mode_object in 'shapes' 'mnist'; do
                for num_objects_all in '2 3' '4'; do
                    name=$mode_object'_'${num_objects_all// /_}
                    python $folder_src'/convert_gray_to_rgb.py' \
                        --name $name \
                        --folder_inputs $folder_inputs \
                        --folder_outputs $folder_outputs \
                        --only_object $only_object \
                        --same_color $same_color
                done
            done
        done
    done
done

# Combine
for occlude in 'sep' 'occ'; do
    for color in 'gray' 'rgb_1' 'rgb_2' 'rgb_3' 'rgb_4'; do
        folder=$occlude'_'$color
        for object in 'shapes' 'mnist'; do
            python $folder_src'/combine.py' \
                --folder $folder \
                --name $object
        done
    done
done
