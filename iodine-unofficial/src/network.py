import sonnet as snt
import tensorflow.compat.v1 as tf
from functools import partial


def get_grid(shape, name='grid'):
    with tf.name_scope(name):
        rows = tf.linspace(-1.0, 1.0, shape[0])
        cols = tf.linspace(-1.0, 1.0, shape[1])
        grid_cols, grid_rows = tf.meshgrid(cols, rows)
        grid = tf.expand_dims(tf.stack([grid_cols, grid_rows], axis=-1), axis=0)
    return grid


class Initializer(snt.AbstractModule):

    def __init__(self, config, name='initializer'):
        super(Initializer, self).__init__(name=name)
        self._latent_size = config['latent_size']
        self._state_size = config['state_size']

    def _build(self, batch_size, num_slots):
        init_loc = tf.get_variable('loc', shape=[1, 1, self._latent_size], initializer=tf.zeros_initializer())
        init_scale = tf.get_variable('scale', shape=[1, 1, self._latent_size], initializer=tf.zeros_initializer())
        with tf.name_scope('post_param'):
            loc = tf.broadcast_to(init_loc, [batch_size, num_slots, self._latent_size])
            raw_scale = tf.broadcast_to(init_scale, [batch_size, num_slots, self._latent_size])
        with tf.name_scope('state'):
            if self._state_size:
                zeros = tf.zeros((batch_size * num_slots, self._state_size))
                states = snt.LSTMState(hidden=zeros, cell=zeros)
            else:
                states = None
        return loc, raw_scale, states


class Updater(snt.AbstractModule):

    def __init__(self, config, name='updater'):
        super(Updater, self).__init__(name=name)
        assert len(config['upd_channel']) == len(config['upd_kernel'])
        assert len(config['upd_channel']) == len(config['upd_stride'])
        with self._enter_variable_scope(check_same_graph=False):
            self._grid = get_grid(config['image_shape'][:-1], name='grid')
            self._layers = []
            for idx, (channel, kernel, stride) in enumerate(
                    zip(config['upd_channel'], config['upd_kernel'], config['upd_stride'])):
                self._layers += [
                    snt.Conv2D(channel, kernel, stride=stride, name='conv_{}'.format(idx)),
                    partial(tf.nn.elu, name='conv_{}_elu'.format(idx)),
                ]
            self._layers.append(partial(tf.math.reduce_mean, axis=[1, 2], name='global_avg_pool'))
            for idx, hidden in enumerate(config['upd_hidden']):
                self._layers += [
                    snt.Linear(hidden, name='linear_{}'.format(idx)),
                    partial(tf.nn.elu, name='linear_{}_elu'.format(idx)),
                ]
            if config['state_size']:
                self._lstm = snt.LSTM(config['state_size'], name='lstm')
            else:
                self._lstm = None
            self._linear_loc = snt.Linear(config['latent_size'], name='linear_loc')
            self._linear_scale = snt.Linear(config['latent_size'], name='linear_scale')

    def _build(self, loc, raw_scale, inputs, states, batch_size):
        with tf.name_scope('broadcast'):
            grid = tf.broadcast_to(self._grid, [batch_size * loc.shape[1], *self._grid.shape[1:]])
        net = tf.concat([inputs[0], grid], axis=-1, name='concat_conv')
        for layer in self._layers:
            net = layer(net)
        net = tf.concat([net, inputs[1]], axis=-1, name='concat_linear')
        if self._lstm:
            net, states = self._lstm(net, states)
        delta_loc = self._linear_loc(net)
        delta_raw_scale = self._linear_scale(net)
        with tf.name_scope('update'):
            delta_loc = tf.reshape(delta_loc, [batch_size, *loc.shape[1:]])
            delta_raw_scale = tf.reshape(delta_raw_scale, [batch_size, *raw_scale.shape[1:]])
            loc = loc + delta_loc
            raw_scale = raw_scale + delta_raw_scale
        return loc, raw_scale, states


class Decoder(snt.AbstractModule):

    def __init__(self, config, name='decoder'):
        super(Decoder, self).__init__(name=name)
        assert len(config['dec_channel']) == len(config['dec_kernel']) - 1
        assert len(config['dec_channel']) == len(config['dec_shape'])
        self._image_shape = config['image_shape']
        self._shapes = [[n, n] if isinstance(n, int) else n for n in config['dec_shape']]
        with self._enter_variable_scope(check_same_graph=False):
            self._grid = get_grid(self._image_shape[:-1], 'grid')
            self._layers = [
                snt.Conv2D(channel, kernel, name='conv_{}'.format(idx))
                for idx, (channel, kernel) in enumerate(zip(config['dec_channel'], config['dec_kernel']))
            ]
            self._conv_out = snt.Conv2D(self._image_shape[-1] + 1, config['dec_kernel'][-1], name='conv_out')

    def _build(self, loc, raw_scale, batch_size):
        with tf.name_scope('sample'):
            scale = tf.math.softplus(raw_scale)
            noise = tf.random.normal((batch_size, *scale.shape[1:]))
            sample = loc + scale * noise
        with tf.name_scope('broadcast'):
            batch_mul_slot = batch_size * sample.shape[1]
            net = tf.reshape(sample, [batch_mul_slot, *sample.shape[2:]])
            net = tf.expand_dims(tf.expand_dims(net, axis=1), axis=2)
            net = tf.broadcast_to(net, [batch_mul_slot, *self._grid.shape[1:3], net.shape[-1]])
            grid = tf.broadcast_to(self._grid, [batch_mul_slot, *self._grid.shape[1:]])
        net = tf.concat([net, grid], axis=-1, name='concat')
        for idx, (layer, shape) in enumerate(zip(self._layers, self._shapes)):
            if net.shape[1:3] != shape:
                net = tf.image.resize_bilinear(net, shape)
            net = layer(net)
            net = tf.nn.elu(net, name='elu_{}'.format(idx))
        if net.shape[1:3] != self._image_shape[:-1]:
            net = tf.image.resize_bilinear(net, self._image_shape[:-1])
        net = self._conv_out(net)
        with tf.name_scope('convert'):
            net = tf.reshape(net, [batch_size, sample.shape[1], *net.shape[1:]])
            apc, logits_mask = tf.split(net, [self._image_shape[-1], 1], axis=-1)
            apc = (apc + 1) * 0.5
            mask = tf.math.softmax(logits_mask, axis=1)
            log_mask = tf.math.log_softmax(logits_mask, axis=1)
        return apc, mask, log_mask, logits_mask
