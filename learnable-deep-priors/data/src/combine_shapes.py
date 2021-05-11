import argparse
import h5py
import os
from common import save_dataset


def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='shapes_28x28')
    args = parser.parse_args()
    # Datasets
    files_in = {key: '{}_{}.h5'.format(args.name, key) for key in [2, 3, 4]}
    images, labels, objects = {}, {}, {}
    with h5py.File(files_in[3], 'r') as f:
        for key in f:
            images[key] = f[key]['image'][()]
            labels[key] = {sub_key: f[key][sub_key][()] for sub_key in ['segment', 'overlap']}
            objects[key] = f[key]['layers'][()]
    for num_objects in [2, 4]:
        with h5py.File(files_in[num_objects], 'r') as f:
            key_in = 'test'
            key_out = 'general_{}'.format(num_objects)
            images[key_out] = f[key_in]['image'][()]
            labels[key_out] = {sub_key: f[key_in][sub_key][()] for sub_key in ['segment', 'overlap']}
            objects[key_out] = f[key_in]['layers'][()]
    save_dataset(args.name, images, labels, objects)
    for file in files_in.values():
        os.remove(file)
    return


if __name__ == '__main__':
    main()
