folder_src='src'
mode='shop_multi'
name='shop_multi_simple'
path_config='config_'$mode'.yaml'
folder_base='./shop-vrb-gen/output_multi/image_samples'
folder_train=$folder_base'/3_6'
folder_general=$folder_base'/7_10'
folder_out='./rgb'
python $folder_src'/create_complex.py' \
    --name $name \
    --path_config $path_config \
    --folder_train $folder_train \
    --folder_general $folder_general \
    --folder_out $folder_out \
    --multiview