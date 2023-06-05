# Time-Conditioned Generative Modeling of Object-Centric Representations for Video Decomposition and Prediction

This is the code repository of the paper "Time-Conditioned Generative Modeling of Object-Centric Representations for Video Decomposition and Prediction".

## Dependence

- pytorch == 1.8
- torchvision == 0.9
- numpy == 1.19
- h5py == 2.10
- pyyaml == 5.4
- tensorboard == 2.5
- scipy == 1.6
- tqdm == 4.64

```
## Datasets

Synthesize images in the CLEVR datasets using  Blender. The tested version of Blender is 2.79b.
Synthesize images in the SHOP datasets using  Blender. The tested version of Blender is 2.83.
```

for CLEVR-SIMPLE

```
cd clevr-dataset-gen/video_generation_simple
bash create_blend.sh
bash create_360pngs.sh
```

for CLEVR-COMPLEX

```
cd clevr-dataset-gen/video_generation_complex
bash create_blend.sh
bash create_360pngs.sh
```

for SHOP-SIMPLE

```
cd shop-vrb-gen/video_generation_simple
bash create_blend.sh
bash create_360pngs.sh
```

for SHOP-COMPLEX

```
cd shop-vrb-gen/video_generation_complex
bash create_blend.sh
bash create_360pngs.sh
```

Generate the h5py file

```
bash create_clevr.sh
bash create_shop.sh
```

## Experiments

```
cd experiments
```

If you want to run the datasets, e.g. SHOP_SIMPLE.  Change the required gpu by setting export CUDA_VISIBLE_DEVICES in run_shop.sh (i.e. export CUDA_VISIBLE_DEVICES == '0' denotes using gpu:0)
Training the dataset need to change the content run_shop.sh

```
#!/bin/bash
export CUDA_VISIBLE_DEVICES='0'
run_file='../src/main.py'
folder_data= '.'  # your data folder
name='shop'
mode='simple'
num_view=10
path_config='../experiments/'$name'_'$mode'/config_first_stage.yaml'.  # config file
path_data=$folder_data'/'$name'_multi/'$name'_multi_'$mode'_'$num_view'.h5'
folder_log='.'  #your log folder
folder_out='.' # your out folder
python $run_file \
        --path_config $path_config \
        --path_data $path_data \
        --folder_log $folder_log \
        --folder_out $folder_out \
        --train
```
