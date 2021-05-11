import sonnet as snt
import tensorflow.compat.v1 as tf
from functools import partial


class UNetBlockDown(snt.AbstractModule):

    def __init__(self, channel, kernel, stride, name='unet_block_down'):
        super(UNetBlockDown, self).__init__(name=name)
        self._stride = stride
        with self._enter_variable_scope(check_same_graph=False):
            self._layers = [
                snt.Conv2D(channel, kernel, use_bias=False, name='conv'),
                snt.LayerNorm(axis=[1, 2], offset=True, scale=False, name='instance_norm'),
                partial(tf.nn.relu, name='relu'),
            ]

    def _build(self, inputs):
        net = inputs
        for layer in self._layers:
            net = layer(net)
        if self._stride == 1:
            net_down = net
        else:
            shape_down = (net.shape[1] // self._stride, net.shape[2] // self._stride)
            net_down = tf.image.resize_nearest_neighbor(net, shape_down, name='down_sample')
        return net_down, net


class UNetBlockUp(snt.AbstractModule):

    def __init__(self, channel, kernel, name='unet_block_up'):
        super(UNetBlockUp, self).__init__(name=name)
        with self._enter_variable_scope(check_same_graph=False):
            self._layers = [
                snt.Conv2D(channel, kernel, use_bias=False, name='conv'),
                snt.LayerNorm(axis=[1, 2], offset=True, scale=False, name='instance_norm'),
                partial(tf.nn.relu, name='relu'),
            ]

    def _build(self, inputs, inputs_skip):
        if inputs.shape[1:3] != inputs_skip.shape[1:3]:
            inputs = tf.image.resize_nearest_neighbor(inputs, inputs_skip.shape[1:3], name='up_sample')
        net = tf.concat([inputs, inputs_skip], axis=-1, name='concat')
        for layer in self._layers:
            net = layer(net)
        return net


class AttentionNet(snt.AbstractModule):

    def __init__(self, config, name='attention_net'):
        super(AttentionNet, self).__init__(name=name)
        assert len(config['att_channel']) == len(config['att_kernel'])
        with self._enter_variable_scope(check_same_graph=False):
            self._layers_down = []
            self._layers_up_rev = []
            plane_ht, plane_wd = config['image_shape'][:2]
            for idx, (channel, kernel, stride) in enumerate(
                    zip(config['att_channel'], config['att_kernel'], config['att_stride'])):
                self._layers_down.append(UNetBlockDown(channel, kernel, stride, name='down_{}'.format(idx)))
                self._layers_up_rev.append(UNetBlockUp(channel, kernel, name='up_{}'.format(idx)))
                plane_ht //= stride
                plane_wd //= stride
            self._conv_out = snt.Conv2D(1, 1, name='conv_out')
            layers_linear = [snt.BatchFlatten(name='flatten')]
            channel_last = config['att_channel'][-1]
            for idx, hidden in enumerate(config['att_hidden'] + [plane_ht * plane_wd * channel_last]):
                layers_linear += [
                    snt.Linear(hidden, name='linear_{}'.format(idx)),
                    partial(tf.nn.relu, name='relu_{}'.format(idx)),
                ]
            layers_linear.append(snt.BatchReshape([plane_ht, plane_wd, channel_last], name='reshape'))
            self._linear = snt.Sequential(layers_linear, name='connect')

    def _build(self, inputs_1, inputs_2):
        inputs_1 = inputs_1 * 2 - 1
        inputs_2 = tf.stop_gradient(inputs_2)
        net = tf.concat([inputs_1, inputs_2], axis=-1, name='concat')
        net_skip_list = []
        for layer in self._layers_down:
            net, net_skip = layer(net)
            net_skip_list.append(net_skip)
        net = self._linear(net)
        for layer, net_skip in zip(reversed(self._layers_up_rev), reversed(net_skip_list)):
            net = layer(net, net_skip)
        logits_att = self._conv_out(net)
        with tf.name_scope('convert'):
            log_att = tf.math.log_sigmoid(logits_att)
            log1m_att = log_att - logits_att
        return log_att, log1m_att


class Encoder(snt.AbstractModule):

    def __init__(self, config, name='encoder'):
        super(Encoder, self).__init__(name=name)
        assert len(config['enc_channel']) == len(config['enc_kernel'])
        assert len(config['enc_channel']) == len(config['enc_stride'])
        with self._enter_variable_scope(check_same_graph=False):
            self._layers = []
            for idx, (channel, kernel, stride) in enumerate(
                    zip(config['enc_channel'], config['enc_kernel'], config['enc_stride'])):
                self._layers += [
                    snt.Conv2D(channel, kernel, stride=stride, name='conv_{}'.format(idx)),
                    partial(tf.nn.relu, name='conv_relu_{}'.format(idx)),
                ]
            self._layers.append(snt.BatchFlatten(name='flatten'))
            for idx, hidden in enumerate(config['enc_hidden']):
                self._layers += [
                    snt.Linear(hidden, name='linear_{}'.format(idx)),
                    partial(tf.nn.relu, name='linear_relu_{}'.format(idx)),
                ]
            self._layers.append(snt.Linear(config['latent_size'] * 2, name='linear_out'))

    def _build(self, inputs_1, inputs_2):
        inputs_1 = inputs_1 * 2 - 1
        inputs_2 = tf.stop_gradient(inputs_2)
        net = tf.concat([inputs_1, inputs_2], axis=-1, name='concat')
        for layer in self._layers:
            net = layer(net)
        with tf.name_scope('convert'):
            loc, raw_scale = tf.split(net, 2, axis=-1)
        return loc, raw_scale


class Decoder(snt.AbstractModule):

    def __init__(self, config, name='decoder'):
        super(Decoder, self).__init__(name=name)
        assert len(config['dec_channel']) == len(config['dec_kernel'])
        assert len(config['dec_channel']) == len(config['dec_shape'])
        with self._enter_variable_scope(check_same_graph=False):
            dec_shape_list = [(n, n) if isinstance(n, int) else n for n in config['dec_shape']]
            plane_ht, plane_wd = dec_shape_list[0]
            with tf.name_scope('grid'):
                rows = tf.linspace(-1.0, 1.0, plane_ht)
                cols = tf.linspace(-1.0, 1.0, plane_wd)
                grid_rows, grid_cols = tf.meshgrid(rows, cols)
                self._grid = tf.expand_dims(tf.stack([grid_cols, grid_rows], axis=-1), axis=0)
            self._layers = []
            for idx, (channel, kernel, shape) in enumerate(
                    zip(config['dec_channel'], config['dec_kernel'], dec_shape_list)):
                if (plane_ht, plane_wd) != shape:
                    self._layers.append(partial(tf.image.resize_bilinear, size=shape, name='resize_{}'.format(idx)))
                self._layers += [
                    snt.Conv2D(channel, kernel, padding='VALID', name='conv_{}'.format(idx)),
                    partial(tf.nn.relu, name='relu_{}'.format(idx)),
                ]
                plane_ht -= kernel - 1
                plane_wd -= kernel - 1
            if [plane_ht, plane_wd] != config['image_shape'][:2]:
                self._layers.append(
                    partial(tf.image.resize_bilinear, size=config['image_shape'][:2], name='resize_out'))
            self._image_ch = config['image_shape'][-1]
            self._layers.append(snt.Conv2D(self._image_ch + 1, 1, name='conv_out'))

    def _build(self, inputs, batch_size):
        with tf.name_scope('broadcast'):
            net = tf.expand_dims(tf.expand_dims(inputs, axis=1), axis=2)
            net = tf.broadcast_to(net, [batch_size, *self._grid.shape[1:3], net.shape[-1]])
            grid = tf.broadcast_to(self._grid, [batch_size, *self._grid.shape[1:]])
        net = tf.concat([net, grid], axis=-1, name='concat')
        for layer in self._layers:
            net = layer(net)
        with tf.name_scope('convert'):
            apc, logits_mask = tf.split(net, [self._image_ch, 1], axis=-1)
            apc = (apc + 1) * 0.5
        return apc, logits_mask


class VAENet(snt.AbstractModule):

    def __init__(self, config, name='vae_net'):
        super(VAENet, self).__init__(name=name)
        with self._enter_variable_scope(check_same_graph=False):
            self.enc = Encoder(config)
            self.dec = Decoder(config)

    def _build(self, inputs_1, inputs_2, batch_size):
        loc, raw_scale = self.enc(inputs_1, inputs_2)
        with tf.name_scope('sample'):
            scale = tf.math.softplus(raw_scale)
            noise = tf.random.normal((batch_size, *scale.shape[1:]))
            sample = loc + scale * noise
        apc, logits_mask = self.dec(sample, batch_size)
        return apc, logits_mask, loc, raw_scale
