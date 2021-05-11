import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import argparse
import tensorflow.compat.v1 as tf
from common import get_next_batch, create_dataset
from multi_object_datasets import objects_room


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_in')
    parser.add_argument('--folder_out')
    parser.add_argument('--num_data', type=int, default=60000)
    parser.add_argument('--num_parallel', type=int, default=32)
    args = parser.parse_args()
    if not os.path.exists(args.folder_out):
        os.makedirs(args.folder_out)
    for split in ['train', 'empty_room', 'six_objects', 'identical_color']:
        if split == 'train':
            file_in = 'objects_room_{}.tfrecords'.format(split)
        else:
            file_in = 'objects_room_test_{}.tfrecords'.format(split)
        path_in = os.path.join(args.folder_in, 'objects_room', file_in)
        dataset = objects_room.dataset(path_in, split, map_parallel_calls=args.num_parallel)
        next_batch = get_next_batch(dataset, args.num_data)
        path_out = os.path.join(args.folder_out, 'room_{}'.format(split))
        with tf.Session() as sess:
            data = sess.run(next_batch)
        create_dataset(data, path_out, args.num_parallel)
    return


if __name__ == '__main__':
    main()
