import argparse
import h5py
import os
import tensorflow as tf
from multiprocessing.pool import Pool


def convert(data):
    def get_bytes_feature(x):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=x))
    feature = {key: get_bytes_feature([n.tobytes() for n in val.flatten()]) for key, val in data.items()}
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()
    return example_proto


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_in')
    parser.add_argument('--folder_out')
    parser.add_argument('--name')
    parser.add_argument('--num_parallel', type=int, default=32)
    args = parser.parse_args()
    folder_out = os.path.join(args.folder_out, args.name)
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    key_list = ['image', 'segment', 'overlap']
    with h5py.File(os.path.join(args.folder_in, '{}.h5'.format(args.name)), 'r') as f:
        for phase in f:
            data = {key: f[phase][key][()] for key in key_list}
            data = [{key: data[key][idx] for key in key_list} for idx in range(data[key_list[0]].shape[0])]
            with Pool(args.num_parallel) as pool:
                data = pool.map(convert, data)
            with tf.io.TFRecordWriter(os.path.join(folder_out, '{}.tfrecord'.format(phase)), 'GZIP') as writer:
                for example_proto in data:
                    writer.write(example_proto)
    return


if __name__ == '__main__':
    main()
