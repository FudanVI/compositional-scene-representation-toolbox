import h5py
import numpy as np
import tensorflow.compat.v1 as tf
from multiprocessing.pool import Pool


def get_next_batch(dataset, batch_size):
    dataset = dataset.batch(batch_size).prefetch(1)
    iterator = tf.data.make_one_shot_iterator(dataset)
    next_batch = iterator.get_next()
    return next_batch


def convert_data(data):
    images = data['image']
    masks = data['mask'].squeeze(-1)
    masks = np.concatenate([masks[:, 1:], masks[:, :1]], axis=1)
    segments = np.argmax(masks, axis=1).astype(np.uint8)
    overlaps = (segments != masks.shape[1] - 1).astype(np.uint8)
    data = {'image': images, 'segment': segments, 'overlap': overlaps}
    return data


def get_example_proto(example):
    def get_bytes_feature(x):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=x))
    feature = {key: get_bytes_feature([n.tobytes() for n in val.flatten()]) for key, val in example.items()}
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()
    return example_proto


def create_dataset(data, path_out, num_parallel):
    data = convert_data(data)
    with h5py.File(path_out + '.h5', 'w') as f:
        for key, val in data.items():
            f.create_dataset(key, data=val, compression='gzip')
    data = [{key: val[idx] for key, val in data.items()} for idx in range(data['image'].shape[0])]
    with Pool(num_parallel) as pool:
        data = pool.map(get_example_proto, data)
    with tf.io.TFRecordWriter(path_out + '.tfrecords', 'GZIP') as writer:
        for example_proto in data:
            writer.write(example_proto)
    return
