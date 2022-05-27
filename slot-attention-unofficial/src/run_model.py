import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from utils import compute_outputs, convert_image, plot_image, add_summary_scalar, init_train


def get_distributed_train_step(strategy, config, net, optimizer):
    def train_step(data, num_slots, require_extra):
        with tf.GradientTape() as tape:
            results, metrics, losses = net(data, num_slots, require_extra)
            loss_opt = tf.reduce_sum(losses['opt']) / config['batch_size']
        gradients = tape.gradient(loss_opt, net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, net.trainable_variables))
        return results, metrics, losses
    def distributed_train_step(data, num_slots, require_extra):
        results, metrics, losses = strategy.run(train_step, args=(data, num_slots, require_extra))
        results, scalars = compute_outputs(strategy, results, metrics, losses)
        return results, scalars
    if config['debug']:
        def fn_distributed_train_step(data, num_slots, require_extra):
            return distributed_train_step(data, num_slots, require_extra)
    else:
        @tf.function
        def fn_distributed_train_step(data, num_slots, require_extra):
            return distributed_train_step(data, num_slots, require_extra)
    return fn_distributed_train_step


def get_distributed_test_step(strategy, config, net):
    def test_step(data, num_slots, require_extra):
        results, metrics, losses = net(data, num_slots, require_extra)
        return results, metrics, losses
    def distributed_test_step(data, num_slots, require_extra):
        results, metrics, losses = strategy.run(test_step, args=(data, num_slots, require_extra))
        results, scalars = compute_outputs(strategy, results, metrics, losses)
        return results, scalars
    if config['debug']:
        def fn_distributed_test_step(data, num_slots, require_extra):
            return distributed_test_step(data, num_slots, require_extra)
    else:
        @tf.function
        def fn_distributed_test_step(data, num_slots, require_extra):
            return distributed_test_step(data, num_slots, require_extra)
    return fn_distributed_test_step


def add_summary_image(summary_writer, config, phase, step, results, dpi=100):
    def get_overview(fig_idx):
        image = results['image'][fig_idx]
        recon = results['recon'][fig_idx]
        apc = results['apc'][fig_idx]
        mask = results['mask'][fig_idx]
        pres = results['pres'][fig_idx]
        num_slots = mask.shape[0]
        num_rows, num_cols = 2, num_slots + 1
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols, num_rows + 0.2), dpi=dpi)
        plot_image(axes[0, 0], convert_image(image), xlabel='scene')
        plot_image(axes[1, 0], convert_image(recon))
        for col in range(num_slots):
            font_color = [1.0, 0.5, 0.0] if pres[col] else [0.0, 0.5, 1.0]
            plot_image(axes[0, col + 1], convert_image(apc[col]), xlabel='slot_{}'.format(col), font_color=font_color)
            plot_image(axes[1, col + 1], convert_image(mask[col]))
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        out = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, -1)[None]
        plt.close(fig)
        return out
    overview_list = [get_overview(idx) for idx in range(config['summ_image_count'])]
    overview = np.concatenate(overview_list, axis=1)
    with summary_writer.as_default():
        tf.summary.image('{}/overview'.format(phase), overview, step=step, max_outputs=1)
    return


def train_model(strategy, config, data_loaders, net, optimizer, checkpoint):
    distributed_train_step = get_distributed_train_step(strategy, config, net, optimizer)
    distributed_valid_step = get_distributed_test_step(strategy, config, net)
    phase_param_train = config['phase_param']['train']
    phase_param_valid = config['phase_param']['valid'] if 'valid' in config['phase_param'] else None
    summary_writer, net = init_train(strategy, config, checkpoint, net)
    step = checkpoint['ckpt'].step.numpy()
    for data_train in data_loaders['train']:
        step += 1
        if step > config['num_steps']:
            break
        _, scalars = distributed_train_step(data_train, phase_param_train['num_slots'], require_extra=False)
        scalars = {key: np.mean(val) for key, val in scalars.items()}
        add_summary_scalar(summary_writer, 'train', step, scalars)
        if step % config['ckpt_intvl'] == 0:
            if phase_param_valid is not None:
                valid_scalars = {}
                for idx_batch, data_valid in enumerate(data_loaders['valid']):
                    require_extra = (idx_batch == 0)
                    results, scalars = distributed_valid_step(data_valid, phase_param_valid['num_slots'], require_extra)
                    for key, val in scalars.items():
                        if key in valid_scalars:
                            valid_scalars[key].append(val)
                        else:
                            valid_scalars[key] = [val]
                    if require_extra:
                        add_summary_image(summary_writer, config, 'valid', step, results)
                valid_scalars = {key: np.mean(np.concatenate(val)) for key, val in valid_scalars.items()}
                add_summary_scalar(summary_writer, 'valid', step, valid_scalars)
                summary_writer.flush()
                if valid_scalars['loss_compare'] < checkpoint['ckpt'].best_loss:
                    checkpoint['ckpt'].best_loss.assign(valid_scalars['loss_compare'])
                    checkpoint['ckpt'].best_step.assign(step)
                    net.save_weights(os.path.join(config['folder_out'], config['folder_model'], config['name_save']))
                checkpoint['ckpt'].step.assign(step)
                checkpoint['ckpt_mgr'].save(checkpoint_number=step)
                print('Step: {}/{}'.format(step, config['num_steps']))
                print((' ' * 4).join([
                    'ARI: {:.3f}'.format(valid_scalars['metric_ari']),
                    'MSE: {:.2e}'.format(valid_scalars['metric_mse']),
                    'Count: {:.3f}'.format(valid_scalars['metric_count']),
                ]))
                print('Best Step: {}'.format(checkpoint['ckpt'].best_step.numpy()))
                print()
            else:
                summary_writer.flush()
                net.save_weights(os.path.join(config['folder_out'], config['folder_model'], config['name_save']))
                checkpoint['ckpt'].step.assign(step)
                checkpoint['ckpt_mgr'].save(checkpoint_number=step)
                print('Step: {}/{}'.format(step, config['num_steps']))
    summary_writer.close()
    return


def test_model(strategy, config, data_loaders, net):
    net.load_weights(os.path.join(config['folder_out'], config['folder_model'], config['name_save']))
    distributed_test_step = get_distributed_test_step(strategy, config, net)
    require_extra = True
    for phase in data_loaders:
        if phase in ['train', 'valid']:
            continue
        phase_param = config['phase_param'][phase]
        num_slots = phase_param['num_slots']
        with h5py.File(os.path.join(config['folder_out'], '{}.h5'.format(phase)), 'w') as f:
            for data in data_loaders[phase]:
                outputs = {key: [] for key in ['recon', 'apc', 'mask', 'pres']}
                for _ in range(config['num_tests']):
                    results, _ = distributed_test_step(data, num_slots, require_extra)
                    for key in outputs:
                        val = results[key]
                        val = tf.cast(tf.clip_by_value(val, 0, 1) * 255, tf.uint8)
                        outputs[key].append(val)
                outputs = {key: tf.stack(val).numpy() for key, val in outputs.items()}
                for key, val in outputs.items():
                    if key in f:
                        f[key].resize(f[key].shape[1] + val.shape[1], axis=1)
                        f[key][:, -val.shape[1]:] = val
                    else:
                        f.create_dataset(
                            key, data=val, maxshape=[val.shape[0], None, *val.shape[2:]], compression='gzip')
    return
