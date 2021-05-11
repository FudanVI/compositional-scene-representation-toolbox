import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
from common import get_next_batch, create_dataset
from multi_object_datasets import clevr_with_masks
from skimage.transform import resize


def crop_and_resize(x):
    x = x[29:221, 64:256]
    x = resize(x, (128, 128), preserve_range=True).astype(np.uint8)
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_in')
    parser.add_argument('--folder_out')
    parser.add_argument('--num_data', type=int, default=60000)
    parser.add_argument('--num_parallel', type=int, default=32)
    args = parser.parse_args()
    if not os.path.exists(args.folder_out):
        os.makedirs(args.folder_out)
    path_in = os.path.join(args.folder_in, 'clevr_with_masks', 'clevr_with_masks_train.tfrecords')
    dataset = clevr_with_masks.dataset(path_in, map_parallel_calls=args.num_parallel)
    next_batch = get_next_batch(dataset, args.num_data)
    path_out = os.path.join(args.folder_out, 'clevr')
    with tf.Session() as sess:
        data = sess.run(next_batch)
    data['image'] = np.stack([crop_and_resize(val) for val in data['image']])
    data['mask'] = np.stack([np.stack([crop_and_resize(sub_val) for sub_val in val]) for val in data['mask']])
    create_dataset(data, path_out, args.num_parallel)
    return


if __name__ == '__main__':
    main()
