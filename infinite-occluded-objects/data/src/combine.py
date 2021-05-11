import argparse
import h5py
import os
from common import save_dataset


def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder')
    parser.add_argument('--name')
    args = parser.parse_args()
    # Datasets
    files_in = {key: os.path.join(args.folder, '{}_{}.h5'.format(args.name, key)) for key in ['2_3', '4']}
    images, labels, objects = {}, {}, {}
    with h5py.File(files_in['2_3'], 'r') as f:
        for key in f:
            images[key] = f[key]['image'][()]
            labels[key] = {sub_key: f[key][sub_key][()] for sub_key in ['segment', 'overlap']}
            objects[key] = f[key]['layers'][()]
    with h5py.File(files_in['4'], 'r') as f:
        key_in = 'test'
        key_out = 'general'
        images[key_out] = f[key_in]['image'][()]
        labels[key_out] = {sub_key: f[key_in][sub_key][()] for sub_key in ['segment', 'overlap']}
        objects[key_out] = f[key_in]['layers'][()]
    save_dataset(os.path.join(args.folder, args.name), images, labels, objects)
    for file in files_in.values():
        os.remove(file)
    return


if __name__ == '__main__':
    main()
