import argparse
import json
import numpy as np
import os
from common import load_config, save_dataset


def generate_data(config, folder):
    num_views = config['num_views']
    data_list = []
    for idx_view in range(num_views):
        with open(os.path.join(folder, 'view_{}.json'.format(idx_view)), 'r') as f:
            data = json.load(f)
        data_list.append([
            np.cos(data['theta']), np.sin(data['theta']),
            np.cos(data['phi']), np.sin(data['phi']),
            data['rho'],
        ])
    data = {'viewpoint': np.array(data_list).astype(np.float32)}
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name')
    parser.add_argument('--path_config')
    parser.add_argument('--folder_train')
    parser.add_argument('--folder_general')
    parser.add_argument('--folder_out')
    config = load_config(parser)
    sub_folders_train = sorted(os.listdir(config['folder_train']))
    sub_folders_general = sorted(os.listdir(config['folder_general']))
    folders_all = {
        'train': (config['folder_train'], sub_folders_train[:config['split_train']]),
        'valid': (config['folder_train'], sub_folders_train[config['split_train']:config['split_valid']]),
        'test': (config['folder_train'], sub_folders_train[config['split_valid']:]),
        'general': (config['folder_general'], sub_folders_general),
    }
    folders_all = {key: [os.path.join(val[0], sub_val) for sub_val in val[1]] for key, val in folders_all.items()}
    datasets = {}
    for phase, folders in folders_all.items():
        data_list = []
        for folder in folders:
            data = generate_data(config, folder)
            data_list.append(data)
        datasets[phase] = data_list
    save_dataset(config, datasets)
    return


if __name__ == '__main__':
    main()
