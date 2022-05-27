import math
import numpy as np
import os
import tensorflow as tf
from matplotlib.colors import hsv_to_rgb


def unpack_row_col(x):
    if isinstance(x, int):
        x_row = x_col = x
    else:
        x_row, x_col = x
    return x_row, x_col


def soft_clip(x, lim):
    assert len(lim) == 2
    assert lim[0] < lim[1]
    scl = 0.5 * (lim[1] - lim[0])
    trs = 0.5 * (lim[1] + lim[0])
    x = tf.tanh(x / scl) * scl + trs
    return x


def batch_broadcast(x, batch_size):
    return tf.broadcast_to(x[None], [batch_size, *x.shape])


def batch_sum(x):
    return tf.reduce_sum(x, axis=[*range(1, x.shape.rank)])


def batch_mean(x):
    return tf.reduce_mean(x, axis=[*range(1, x.shape.rank)])


def get_checkpoint(config, net, optimizer):
    step = tf.Variable(0, trainable=False)
    best_step = tf.Variable(-1, trainable=False)
    best_loss = tf.Variable(tf.float32.max, trainable=False)
    ckpt = tf.train.Checkpoint(
        model=net, optimizer=optimizer, step=step, best_step=best_step, best_loss=best_loss)
    ckpt_mgr = tf.train.CheckpointManager(
        ckpt, os.path.join(config['folder_out'], config['folder_ckpt']), max_to_keep=1)
    folder_model = os.path.join(config['folder_out'], config['folder_model'])
    checkpoint = {'ckpt': ckpt, 'ckpt_mgr': ckpt_mgr, 'folder_model': folder_model}
    return checkpoint


def compute_loss_coef(config, loss_coef_init, step):
    loss_coef = {**loss_coef_init}
    for key, val in config['loss_coef'].items():
        step_list = [1] + val['step'] + [config['num_steps']]
        assert len(step_list) == len(val['value'])
        assert len(step_list) == len(val['linear']) + 1
        assert step_list == sorted(step_list)
        for idx in range(len(step_list) - 1):
            if step <= step_list[idx + 1]:
                ratio = (step - step_list[idx]) / (step_list[idx + 1] - step_list[idx])
                val_1 = val['value'][idx]
                val_2 = val['value'][idx + 1]
                if val['linear'][idx]:
                    loss_coef[key] = (1 - ratio) * val_1 + ratio * val_2
                else:
                    loss_coef[key] = math.exp((1 - ratio) * math.log(val_1) + ratio * math.log(val_2))
                assert math.isfinite(loss_coef[key])
                break
        else:
            raise ValueError
    loss_coef = {key: tf.constant(val, dtype=tf.float32) for key, val in loss_coef.items()}
    return loss_coef


def compute_outputs(strategy, results, metrics, losses):
    def reduce_concat(x):
        return {key: tf.concat(val.values, axis=0) for key, val in x.items()}
    if strategy.num_replicas_in_sync > 1:
        results = reduce_concat(results)
        metrics = reduce_concat(metrics)
        losses = reduce_concat(losses)
    scalars = {
        **{'metric_{}'.format(key): val for key, val in metrics.items()},
        **{'loss_{}'.format(key): val for key, val in losses.items()},
    }
    return results, scalars


def convert_image(image):
    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    return image


def combine_mask(mask):
    mask = mask.numpy()
    num_colors = mask.shape[0]
    hsv_colors = np.ones((num_colors, 3))
    hsv_colors[:, 0] = (np.linspace(0, 1, num_colors, endpoint=False) + 2 / 3) % 1.0
    mask_colors = hsv_to_rgb(hsv_colors)
    mask_smry = np.clip((mask * mask_colors[:, None, None]).sum(0), 0, 1)
    return mask_smry, mask_colors


def color_spines(ax, color, lw=3):
    for loc in ['top', 'bottom', 'left', 'right']:
        ax.spines[loc].set_linewidth(lw)
        ax.spines[loc].set_color(color)
        ax.spines[loc].set_visible(True)
    return


def plot_image(ax, image, xlabel=None, ylabel=None, font_color=None, border_color=None):
    plot = ax.imshow(image, interpolation='bilinear')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(xlabel, color='k' if font_color is None else font_color, fontfamily='monospace') if xlabel else None
    ax.set_ylabel(ylabel, color='k' if font_color is None else font_color, fontfamily='monospace') if ylabel else None
    ax.xaxis.set_label_position('top')
    if border_color is not None:
        color_spines(ax, color=border_color)
    return plot


def add_summary_scalar(summary_writer, phase, step, scalars):
    with summary_writer.as_default():
        for key, val in scalars.items():
            tf.summary.scalar('{}/{}'.format(phase, key), val, step=step)
    return


def init_train(strategy, config, checkpoint, net):
    summary_writer = tf.summary.create_file_writer(config['folder_log'])
    if config['resume']:
        with strategy.scope():
            checkpoint['ckpt_mgr'].restore_or_initialize()
        print('Resume training from step {}'.format(checkpoint['ckpt'].step.numpy()))
    else:
        assert checkpoint['ckpt_mgr'].latest_checkpoint is None
        assert not os.path.exists(checkpoint['folder_model'])
        if config['path_pretrain'] is not None:
            net.load_weights(config['path_pretrain'])
        print('Start training')
    return summary_writer, net


def distribute_tensor(strategy, x):
    def value_fn(ctx):
        offset = ctx.replica_id_in_sync_group * batch_size
        return x[offset:offset + batch_size]
    assert x.shape[0] % strategy.num_replicas_in_sync == 0
    batch_size = x.shape[0] // strategy.num_replicas_in_sync
    return strategy.experimental_distribute_values_from_function(value_fn)
