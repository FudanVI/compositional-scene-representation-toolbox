import os
import tensorflow as tf


def get_dataset(config, phase, phase_param):
    def decode(example_proto):
        example = tf.io.parse_single_example(example_proto, features)
        example = {key: tf.squeeze(tf.io.decode_raw(val, tf.uint8), axis=-1) for key, val in example.items()}
        example['overlap'] = tf.greater(example['overlap'], 1)
        return example
    def convert(example):
        data = {
            'image': tf.image.convert_image_dtype(example['image'], tf.float32),
            'segment': tf.one_hot(example['segment'], phase_param['data_slots'], axis=0),
            'overlap': tf.cast(example['overlap'], tf.float32)[None],
        }
        return data
    image_shape = config['image_shape']
    features = {
        'image': tf.io.FixedLenFeature(image_shape, tf.string),
        'segment': tf.io.FixedLenFeature([*image_shape[:-1], 1], tf.string),
        'overlap': tf.io.FixedLenFeature([*image_shape[:-1], 1], tf.string),
    }
    path_list = [os.path.join(config['folder_data'], n) for n in phase_param['data_filenames']]
    dataset = tf.data.TFRecordDataset(path_list, 'GZIP')
    dataset = dataset.map(decode, num_parallel_calls=config['num_parallel'])
    if config['debug']:
        dataset = dataset.take(config['batch_size'])
    if config['data_cache'] and phase in ['train', 'valid']:
        dataset = dataset.cache()
    if phase == 'train':
        dataset = dataset.repeat()
        if config['data_shuffle'] > 0:
            dataset = dataset.shuffle(config['data_shuffle'])
    dataset = dataset.map(convert, num_parallel_calls=config['num_parallel'])
    if phase == 'train':
        dataset = dataset.batch(config['batch_size'], drop_remainder=True)
    else:
        dataset = dataset.batch(config['batch_size'] // 2, drop_remainder=False)
    dataset = dataset.prefetch(1)
    return dataset


def get_data_loaders(strategy, config):
    data_loaders = {key: get_dataset(config, key, val) for key, val in config['phase_param'].items()}
    data_loaders = {key: strategy.experimental_distribute_dataset(val) for key, val in data_loaders.items()}
    return data_loaders
