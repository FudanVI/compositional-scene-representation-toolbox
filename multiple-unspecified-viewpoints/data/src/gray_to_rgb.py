import argparse
import numpy as np
import h5py
import os
from common import convert_layers, save_dataset


def add_color(layers_prev, min_diff_sq):
    layers = np.empty((*layers_prev.shape[:-1], 4), dtype=layers_prev.dtype)
    layers[..., -1] = layers_prev[..., -1]
    color_bck = np.random.rand()
    layers[-1, ..., :-1] = color_bck * 255
    for idx in range(layers.shape[0] - 1):
        while True:
            color_obj = np.random.rand(1, 1, 3)
            if np.square(color_obj - color_bck).sum() >= min_diff_sq:
                break
        layers[idx, ..., :-1] = color_obj * layers_prev[idx, ..., :-1]
    return layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name')
    parser.add_argument('--path_in')
    parser.add_argument('--folder_out')
    parser.add_argument('--min_diff_sq', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    config = args.__dict__
    if not os.path.exists(config['folder_out']):
        os.makedirs(config['folder_out'])
    np.random.seed(config['seed'])
    datasets = {}
    with h5py.File(config['path_in'], 'r') as f:
        for phase in f:
            layers_prev_all = f[phase]['layers'][()]
            data_list = []
            for layers_prev in layers_prev_all:
                layers = add_color(layers_prev, config['min_diff_sq'])
                image, segment, overlap = convert_layers(layers)
                data = {'image': image, 'segment': segment, 'overlap': overlap, 'layers': layers}
                data_list.append(data)
            datasets[phase] = data_list
    save_dataset(config, datasets)
    return


if __name__ == '__main__':
    main()
