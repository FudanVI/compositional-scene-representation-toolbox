import argparse
import h5py
import os
from create_mnist import save_dataset


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_out')
    parser.add_argument('--name_in_1')
    parser.add_argument('--name_in_2')
    args = parser.parse_args()
    # Datasets
    images, labels, objects, classes = {}, {}, {}, {}
    with h5py.File('{}.h5'.format(args.name_in_1), 'r') as f:
        for key in f:
            images[key] = f[key]['image'][()]
            labels[key] = {sub_key: f[key][sub_key][()] for sub_key in ['segment', 'overlap']}
            objects[key] = f[key]['layers'][()]
            classes[key] = f[key]['classes'][()]
    with h5py.File('{}.h5'.format(args.name_in_2), 'r') as f:
        key_in = 'test'
        key_out = 'general'
        images[key_out] = f[key_in]['image'][()]
        labels[key_out] = {sub_key: f[key_in][sub_key][()] for sub_key in ['segment', 'overlap']}
        objects[key_out] = f[key_in]['layers'][()]
        classes[key_out] = f[key_in]['classes'][()]
    save_dataset(args.name_out, images, labels, objects, classes)
    for name in [args.name_in_1, args.name_in_2]:
        os.remove('{}.h5'.format(name))
