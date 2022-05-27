import os
import tensorflow.compat.v1 as tf


def get_dataset(config, phase_param, phase, name='dataset'):
    def decode(example_proto):
        example = tf.parse_single_example(example_proto, features)
        example = {key: tf.squeeze(tf.decode_raw(val, tf.uint8), axis=-1) for key, val in example.items()}
        return example
    def convert(example):
        image = tf.image.convert_image_dtype(example['image'], tf.float32)
        segment = tf.one_hot(example['segment'], phase_param['data_slots'], axis=0)
        overlap = tf.cast(tf.greater(example['overlap'][None], 1), tf.float32)
        data = {'image': image, 'segment': segment, 'overlap': overlap}
        return data
    with tf.name_scope(name):
        image_shape = config['image_shape']
        features = {
            'image': tf.FixedLenFeature(image_shape, tf.string),
            'segment': tf.FixedLenFeature([*image_shape[:-1], 1], tf.string),
            'overlap': tf.FixedLenFeature([*image_shape[:-1], 1], tf.string),
        }
        path_list = [os.path.join(config['folder_data'], n) for n in phase_param['data_filenames']]
        dataset = tf.data.TFRecordDataset(path_list, 'GZIP')
        dataset = dataset.map(decode, num_parallel_calls=config['num_parallel'])
        if config['data_cache'] and phase in ['train', 'valid']:
            dataset = dataset.cache()
        if phase == 'train':
            dataset = dataset.repeat()
            if config['data_shuffle'] > 0:
                dataset = dataset.shuffle(config['data_shuffle'])
        dataset = dataset.map(convert, num_parallel_calls=config['num_parallel'])
        if phase == 'train':
            dataset = dataset.batch(config['batch_size'], drop_remainder=True)
        elif phase == 'valid':
            dataset = dataset.batch(config['batch_size_valid'], drop_remainder=True)
        else:
            dataset = dataset.batch(config['batch_size'] // 2, drop_remainder=False)
        dataset = dataset.prefetch(1)
    return dataset


def get_data_loaders(strategy, config):
    data_loaders = {}
    for key, val in config['phase_param'].items():
        with tf.name_scope('data_{}'.format(key)):
            dataset = get_dataset(config, val, key)
            with tf.name_scope('dist_dataset'):
                dataset = strategy.experimental_distribute_dataset(dataset)
            with tf.name_scope('iterator'):
                data_loaders[key] = tf.data.make_initializable_iterator(dataset)
    return data_loaders
