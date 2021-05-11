import argparse
import h5py
import numpy as np
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name')
    parser.add_argument('--folder_in')
    parser.add_argument('--folder_out')
    args = parser.parse_args()
    if not os.path.exists(args.folder_out):
        os.makedirs(args.folder_out)
    path_in = os.path.join(args.folder_in, '{}.h5'.format(args.name))
    path_out = os.path.join(args.folder_out, '{}.h5'.format(args.name))
    with h5py.File(path_in, 'r') as f_in, h5py.File(path_out, 'w') as f_out:
        for key_in in f_in:
            features = np.rollaxis(f_in[key_in]['features'][()], 0, 2)
            groups = np.rollaxis(f_in[key_in]['groups'][()], 0, 2)
            overlaps = np.rollaxis(f_in[key_in]['overlaps'][()], 0, 2)
            images = (features * 255).astype(np.uint8)
            segments = groups.astype(np.uint8).squeeze(-1)
            overlaps = overlaps.astype(np.uint8).squeeze(-1)
            seg_back_val = segments.max()
            seg_back_pos = segments == 0
            segments -= 1
            segments[seg_back_pos] = seg_back_val
            key_out = key_in[:5]
            f_out.create_group(key_out)
            f_out[key_out].create_dataset('image', data=images, compression='gzip')
            f_out[key_out].create_dataset('segment', data=segments, compression='gzip')
            f_out[key_out].create_dataset('overlap', data=overlaps, compression='gzip')
    return


if __name__ == '__main__':
    main()
