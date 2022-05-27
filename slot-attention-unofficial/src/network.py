import numpy as np
import tensorflow as tf


class NetworkGrid(tf.keras.layers.Layer):

    def __init__(self, units):
        super(NetworkGrid, self).__init__()
        self.net = tf.keras.layers.Dense(units)

    def call(self, x):
        rows = tf.linspace(0.0, 1.0, x.shape[1])
        cols = tf.linspace(0.0, 1.0, x.shape[2])
        grid_rows, grid_cols = tf.meshgrid(rows, cols, indexing='ij')
        grid = tf.stack([grid_rows, grid_cols], axis=-1)[None]
        grid = tf.concat([grid, 1 - grid], axis=-1)
        x += self.net(grid)
        return x


class SlotAttention(tf.keras.layers.Layer):

    def __init__(self, num_steps, slot_size, units_list, eps=1e-8):
        super(SlotAttention, self).__init__()
        self.num_steps = num_steps
        self.slot_size = slot_size
        self.eps = eps
        self.layer_norm_in = tf.keras.layers.LayerNormalization()
        self.layer_norm_qry = tf.keras.layers.LayerNormalization()
        self.layer_norm_linear = tf.keras.layers.LayerNormalization()
        shape = [1, 1, slot_size]
        self.loc = self.add_weight(name='loc', shape=shape, dtype=tf.float32, initializer='glorot_uniform')
        self.log_scl = self.add_weight(name='log_scl', shape=shape, dtype=tf.float32, initializer='glorot_uniform')
        self.net_qry = tf.keras.layers.Dense(slot_size, use_bias=False)
        self.net_key = tf.keras.layers.Dense(slot_size, use_bias=False)
        self.net_val = tf.keras.layers.Dense(slot_size, use_bias=False)
        self.gru = tf.keras.layers.GRUCell(slot_size)
        linear_list = []
        for units in units_list:
            linear_list.append(tf.keras.layers.Dense(units, activation='relu'))
        linear_list.append(tf.keras.layers.Dense(slot_size))
        self.net_linear = tf.keras.Sequential(linear_list)

    def call(self, x, num_slots):
        x = self.layer_norm_in(x)
        x_key = self.net_key(x)
        x_val = self.net_val(x)
        slots = self.loc + tf.exp(self.log_scl) * tf.random.normal([x.shape[0], num_slots, self.slot_size])
        for _ in range(self.num_steps):
            slots_prev = slots
            slots = self.layer_norm_qry(slots)
            x_qry = self.net_qry(slots) / np.sqrt(self.slot_size).astype(np.float32)
            attn_logits = tf.keras.backend.batch_dot(x_key, x_qry, axes=-1)
            attn = tf.math.softmax(attn_logits, axis=-1) + self.eps
            attn /= tf.reduce_sum(attn, axis=-2, keepdims=True)
            updates = tf.keras.backend.batch_dot(attn, x_val, axes=-2)
            slots, _ = self.gru(updates, [slots_prev])
            slots += self.net_linear(self.layer_norm_linear(slots))
        return slots


class Encoder(tf.keras.layers.Layer):

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.net_conv = self.get_conv_block(config)
        self.net_grid = NetworkGrid(config['enc_filters'][-1])
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.net_linear = self.get_linear_block(config)
        self.slot_attn = SlotAttention(
            num_steps=config['slot_steps'],
            slot_size=config['slot_size'],
            units_list=config['enc_slot_units'],
        )

    def call(self, x, num_slots):
        x = x * 2 - 1
        x = self.net_conv(x)
        x = self.net_grid(x)
        x = tf.reshape(x, [x.shape[0], x.shape[1] * x.shape[2], x.shape[3]])
        x = self.layer_norm(x)
        x = self.net_linear(x)
        x = self.slot_attn(x, num_slots)
        return x

    @staticmethod
    def get_conv_block(config):
        filters_list = config['enc_filters']
        kernels_list = config['enc_kernels']
        strides_list = config['enc_strides']
        assert len(filters_list) == len(kernels_list) == len(strides_list)
        conv_list = []
        for filters, kernels, strides in zip(filters_list, kernels_list, strides_list):
            conv_list.append(tf.keras.layers.Conv2D(
                filters, kernel_size=kernels, strides=strides, padding='same', activation='relu'))
        conv_block = tf.keras.Sequential(conv_list)
        return conv_block

    @staticmethod
    def get_linear_block(config):
        units_list = config['enc_main_units']
        linear_list = []
        for idx, units in enumerate(units_list):
            activation = 'relu' if idx < len(units_list) - 1 else None
            linear_list.append(tf.keras.layers.Dense(units, activation=activation))
        linear_block = tf.keras.Sequential(linear_list)
        return linear_block


class Decoder(tf.keras.layers.Layer):

    def __init__(self, config):
        super(Decoder, self).__init__()
        self.feat_shape = config['feat_shape']
        self.net_grid = NetworkGrid(config['slot_size'])
        self.net_conv = self.get_conv_block(config)

    def call(self, x):
        batch_size, num_slots = x.shape[:2]
        x = tf.reshape(x, [-1, *x.shape[2:]])
        x = tf.broadcast_to(x[:, None, None], [x.shape[0], *self.feat_shape, x.shape[1]])
        x = self.net_grid(x)
        x = self.net_conv(x)
        x = tf.reshape(x, [batch_size, num_slots, *x.shape[1:]])
        apc = (x[..., :-1] + 1) * 0.5
        mask = tf.math.softmax(x[..., -1:], axis=1)
        return apc, mask, x[..., -1:]

    @staticmethod
    def get_conv_block(config):
        filters_list = config['dec_filters'] + [config['image_shape'][-1] + 1]
        kernels_list = config['dec_kernels']
        strides_list = config['dec_strides']
        assert len(filters_list) == len(kernels_list) == len(strides_list)
        conv_list = []
        for idx, (filters, kernels, strides) in enumerate(zip(filters_list, kernels_list, strides_list)):
            activation = 'relu' if idx < len(filters_list) - 1 else None
            conv_list.append(tf.keras.layers.Conv2DTranspose(
                filters, kernel_size=kernels, strides=strides, padding='same', activation=activation))
        conv_block = tf.keras.Sequential(conv_list)
        return conv_block
