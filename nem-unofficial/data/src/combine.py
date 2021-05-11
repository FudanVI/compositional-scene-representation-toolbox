import argparse
import h5py


def save_dataset(name, images, labels):
    with h5py.File('{}.h5'.format(name), 'w') as f:
        for key in images:
            f.create_group(key)
            f[key].create_dataset('image', data=images[key], compression='gzip')
            f[key].create_dataset('segment', data=labels[key]['segment'], compression='gzip')
            f[key].create_dataset('overlap', data=labels[key]['overlap'], compression='gzip')
    return


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_out')
    parser.add_argument('--name_in_1')
    parser.add_argument('--name_in_2')
    args = parser.parse_args()
    # Datasets
    images, labels, objects = {}, {}, {}
    with h5py.File('{}.h5'.format(args.name_in_1), 'r') as f:
        for key in f:
            images[key] = f[key]['image'][()]
            labels[key] = {sub_key: f[key][sub_key][()] for sub_key in ['segment', 'overlap']}
    with h5py.File('{}.h5'.format(args.name_in_2), 'r') as f:
        key_in = 'test'
        key_out = 'general'
        images[key_out] = f[key_in]['image'][()]
        labels[key_out] = {sub_key: f[key_in][sub_key][()] for sub_key in ['segment', 'overlap']}
    save_dataset(args.name_out, images, labels)
