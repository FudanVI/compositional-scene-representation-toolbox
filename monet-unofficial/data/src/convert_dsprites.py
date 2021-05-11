import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import argparse
import tensorflow.compat.v1 as tf
from common import get_next_batch, create_dataset
from multi_object_datasets import multi_dsprites


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_in')
    parser.add_argument('--folder_out')
    parser.add_argument('--num_data', type=int, default=60000)
    parser.add_argument('--num_parallel', type=int, default=32)
    args = parser.parse_args()
    if not os.path.exists(args.folder_out):
        os.makedirs(args.folder_out)
    path_in = os.path.join(args.folder_in, 'multi_dsprites', 'multi_dsprites_colored_on_colored.tfrecords')
    dataset = multi_dsprites.dataset(path_in, 'colored_on_colored', map_parallel_calls=args.num_parallel)
    next_batch = get_next_batch(dataset, args.num_data)
    path_out = os.path.join(args.folder_out, 'dsprites')
    with tf.Session() as sess:
        data = sess.run(next_batch)
    create_dataset(data, path_out, args.num_parallel)
    return


if __name__ == '__main__':
    main()
