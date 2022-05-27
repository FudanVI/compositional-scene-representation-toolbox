import argparse
import numpy as np
import h5py
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name')
    parser.add_argument('--path_in')
    parser.add_argument('--folder_out')
    args = parser.parse_args()
    config = args.__dict__
    if not os.path.exists(config['folder_out']):
        os.makedirs(config['folder_out'])
    datasets = {}
    with h5py.File(config['path_in'], 'r') as f:
        for phase in f:
            if 'layers' in f[phase]:
                datasets[phase] = {key: f[phase][key][()] for key in f[phase] if key != 'layers'}
                datasets[phase]['masks'] = f[phase]['layers'][..., -1:]
            else:
                datasets[phase] = {key: f[phase][key][()] for key in f[phase]}
    with h5py.File(os.path.join(config['folder_out'], '{}.h5'.format(config['name'])), 'w') as f:
        for phase in datasets:
            f.create_group(phase)
            for key, val in datasets[phase].items():
                f[phase].create_dataset(key, data=val, compression=True)
    return


if __name__ == '__main__':
    main()
