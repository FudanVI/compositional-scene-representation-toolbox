import gzip
import h5py
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import tensorflow.compat.v1 as tf


def get_loss_coef(config, step):
    loss_coef = {}
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
    return loss_coef


def get_outputs(strategy, op_results, op_metrics, op_losses, name='outputs'):
    def reduce_concat(x):
        return {key: tf.concat(val.values, axis=0) for key, val in x.items()}
    with tf.name_scope(name):
        if strategy.num_replicas_in_sync > 1:
            op_results = reduce_concat(op_results)
            op_metrics = reduce_concat(op_metrics)
            op_losses = reduce_concat(op_losses)
        op_scalars = {
            **{'metric_{}'.format(key): val for key, val in op_metrics.items()},
            **{'loss_{}'.format(key): val for key, val in op_losses.items()},
        }
        ph_summ_image = tf.placeholder(tf.uint8)
        ph_summ_scalar = {key: tf.placeholder(tf.float32) for key in op_scalars}
        op_summ_image = tf.summary.image('overview', ph_summ_image)
        op_summ_scalar = tf.summary.merge([tf.summary.scalar(key, val) for key, val in ph_summ_scalar.items()])
        ops = {'results': op_results, 'scalars': op_scalars, 'summ_image': op_summ_image, 'summ_scalar': op_summ_scalar}
        phs = {'summ_image': ph_summ_image, 'summ_scalar': ph_summ_scalar}
    return ops, phs


def get_train_ops(strategy, config, data_loaders, net, optimizer, phase='train'):
    def get_replica(images, segments, overlaps):
        with tf.GradientTape() as tape:
            results, metrics, losses = net(images, segments, overlaps, phase_param['num_slots'])
            with tf.name_scope('loss_opt'):
                loss_opt = [phs_loss[key] * val for key, val in losses.items() if key != 'compare']
                loss_opt = tf.stack(loss_opt)
                loss_opt = tf.math.reduce_sum(loss_opt) / batch_size_global
        with tf.name_scope('optimization'):
            gradients = tape.gradient(loss_opt, net.trainable_variables)
            gradients = [tf.where(tf.math.is_finite(n), n, tf.zeros_like(n)) for n in gradients]
            optimization = optimizer.apply_gradients(zip(gradients, net.trainable_variables))
        return results, metrics, losses, optimization
    with tf.name_scope('{}_ops'.format(phase)):
        with tf.name_scope('data'):
            data = data_loaders['train'].get_next()
        phase_param = config['phase_param']['train']
        batch_size_global = config['batch_size']
        phs_loss = {key: tf.placeholder(tf.float32, shape=[], name=key) for key in config['loss_coef']}
        op_results, op_metrics, op_losses, op_optimization = \
            strategy.experimental_run_v2(get_replica, args=(data['image'], data['segment'], data['overlap']))
        ops, phs = get_outputs(strategy, op_results, op_metrics, op_losses)
        ops['optimization'] = op_optimization
    return ops, phs, phs_loss


def get_valid_ops(strategy, config, data_loaders, net, phase='valid'):
    def get_replica(images, segments, overlaps):
        results, metrics, losses = net(images, segments, overlaps, phase_param['num_slots'])
        return results, metrics, losses
    with tf.name_scope('{}_ops'.format(phase)):
        with tf.name_scope('data'):
            data = data_loaders['valid'].get_next()
        phase_param = config['phase_param']['valid']
        op_results, op_metrics, op_losses = \
            strategy.experimental_run_v2(get_replica, args=(data['image'], data['segment'], data['overlap']))
        ops, phs = get_outputs(strategy, op_results, op_metrics, op_losses)
    return ops, phs


def add_summary_image(sess, summary_writer, config, step, ops, phs, results, dpi=150):
    def convert_image(image):
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        return image
    def plot_image(ax, image, xlabel=None, ylabel=None, color=None):
        plot = ax.imshow(image, interpolation='bilinear')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(xlabel, color='k' if color is None else color, fontfamily='monospace') if xlabel else None
        ax.set_ylabel(ylabel, color='k' if color is None else color, fontfamily='monospace') if ylabel else None
        ax.xaxis.set_label_position('top')
        return plot
    def get_overview(fig_idx):
        image = results['image'][fig_idx]
        recon = results['recon'][fig_idx]
        recon_att = results['recon_att'][fig_idx]
        apc = results['apc'][fig_idx]
        mask = results['mask'][fig_idx]
        pres = results['pres'][fig_idx]
        mask_att = results['mask_att'][fig_idx]
        pres_att = results['pres_att'][fig_idx]
        rows, cols = 3, apc.shape[0] + 1
        fig, axes = plt.subplots(rows, cols, figsize=(cols, rows + 0.2), dpi=dpi)
        plot_image(axes[0, 0], convert_image(image), xlabel='scene')
        plot_image(axes[1, 0], convert_image(recon))
        plot_image(axes[2, 0], convert_image(recon_att))
        for idx in range(apc.shape[0]):
            plot_image(axes[0, idx + 1], convert_image(apc[idx]), xlabel='obj_{}'.format(idx))
            color = [1.0, 0.5, 0.0] if pres[idx] else [0.0, 0.5, 1.0]
            plot_image(axes[1, idx + 1], convert_image(mask[idx]), color=color)
            color = [1.0, 0.5, 0.0] if pres_att[idx] else [0.0, 0.5, 1.0]
            plot_image(axes[2, idx + 1], convert_image(mask_att[idx]), color=color)
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        out = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, -1)[None]
        plt.close(fig)
        return out
    overview_list = [get_overview(idx) for idx in range(config['summ_image_count'])]
    overview = np.concatenate(overview_list, axis=1)
    summ_image = sess.run(ops['summ_image'], feed_dict={phs['summ_image']: overview})
    summary_writer.add_summary(summ_image, step)
    return


def add_summary_scalar(sess, summary_writer, step, ops, phs, scalars):
    summ_scalar = sess.run(
        ops['summ_scalar'], feed_dict={phs['summ_scalar'][key]: val for key, val in scalars.items()})
    summary_writer.add_summary(summ_scalar, step)
    return


def train_model(strategy, config, data_loaders, net, phase_train='train', phase_valid='valid'):
    def initialize():
        if config['resume']:
            with open(path_ckpt_step, 'rb') as f:
                ckpt_step = pickle.load(f)
            out_step = ckpt_step['step']
            out_best_step = ckpt_step['best_step']
            out_best_loss = ckpt_step['best_loss']
            path_ckpt_param = tf.train.latest_checkpoint(config['folder_out'], latest_filename=config['ckpt_curr'])
            saver_curr.restore(sess, path_ckpt_param)
            print('Resume training from step {}'.format(out_step))
        else:
            out_step = 0
            out_best_step = -1
            out_best_loss = float('inf')
            sess.run(tf.global_variables_initializer())
            print('Start training')
        print()
        return out_step, out_best_step, out_best_loss
    def step_train(ops, phs):
        loss_coef = get_loss_coef(config, step)
        feed_dict = {phs_loss[key]: loss_coef[key] for key in phs_loss}
        scalars, _ = sess.run([ops['scalars'], ops['optimization']], feed_dict=feed_dict)
        scalars = {key: np.mean(val) if key != 'grad_norm' else val for key, val in scalars.items()}
        add_summary_scalar(sess, summary_writer, step, ops, phs, scalars)
        return
    def epoch_valid(ops, phs):
        sess.run(data_loaders[phase_valid].initializer)
        results, scalars = sess.run([ops['results'], ops['scalars']])
        epoch_scalars = {key: [val] for key, val in scalars.items()}
        add_summary_image(sess, summary_writer, config, step, ops, phs, results)
        while True:
            try:
                scalars = sess.run(ops['scalars'])
                for key, val in scalars.items():
                    epoch_scalars[key].append(val)
            except tf.errors.OutOfRangeError:
                break
        epoch_scalars = {key: np.mean(np.concatenate(val)) for key, val in epoch_scalars.items()}
        add_summary_scalar(sess, summary_writer, step, ops, phs, epoch_scalars)
        return epoch_scalars
    optimizer = tf.train.RMSPropOptimizer(learning_rate=config['lr'])
    ops_train, phs_train, phs_loss = get_train_ops(strategy, config, data_loaders, net, optimizer, phase=phase_train)
    use_valid = phase_valid in data_loaders
    if use_valid:
        ops_valid, phs_valid = get_valid_ops(strategy, config, data_loaders, net, phase=phase_valid)
    else:
        ops_valid, phs_valid = None, None
    net_param = list(net.get_all_variables())
    optimizer_param = optimizer.variables()
    saver_curr = tf.train.Saver(net_param + optimizer_param, max_to_keep=1, name='saver_curr')
    saver_best = tf.train.Saver(net_param, max_to_keep=1, name='saver_best')
    path_ckpt_step = os.path.join(config['folder_out'], 'ckpt_step.pickle')
    with tf.Session() as sess:
        with tf.summary.FileWriter(config['folder_log']) as summary_writer:
            step, best_step, best_loss = initialize()
            sess.run(data_loaders[phase_train].initializer)
            while step < config['num_steps']:
                step += 1
                step_train(ops_train, phs_train)
                if step % config['ckpt_intvl'] == 0:
                    if use_valid:
                        valid_scalars = epoch_valid(ops_valid, phs_valid)
                        summary_writer.flush()
                        if valid_scalars['loss_compare'] < best_loss:
                            best_loss = valid_scalars['loss_compare']
                            best_step = step
                            saver_best.save(sess, os.path.join(config['folder_out'], 'best'), global_step=step,
                                            latest_filename=config['ckpt_best'], write_meta_graph=False)
                        saver_curr.save(sess, os.path.join(config['folder_out'], 'curr'), global_step=step,
                                        latest_filename=config['ckpt_curr'], write_meta_graph=False)
                        with open(path_ckpt_step, 'wb') as f:
                            pickle.dump({'step': step, 'best_step': best_step, 'best_loss': best_loss}, f)
                        print('Step: {}/{}'.format(step, config['num_steps']))
                        print((' ' * 4).join([
                            'ARI_A: {:.3f}'.format(valid_scalars['metric_ari_all']),
                            'ARI_O: {:.3f}'.format(valid_scalars['metric_ari_obj']),
                            'MSE: {:.2e}'.format(valid_scalars['metric_mse']),
                            'LL: {:.1f}'.format(valid_scalars['metric_ll']),
                            'Count: {:.3f}'.format(valid_scalars['metric_count']),
                        ]))
                        print((' ' * 4).join([
                            'ARI_A: {:.3f}'.format(valid_scalars['metric_ari_all_att']),
                            'ARI_O: {:.3f}'.format(valid_scalars['metric_ari_obj_att']),
                            'MSE: {:.2e}'.format(valid_scalars['metric_mse_att']),
                            'LL: {:.1f}'.format(valid_scalars['metric_ll_att']),
                            'Count: {:.3f}'.format(valid_scalars['metric_count_att']),
                        ]))
                        print('Best Step: {}'.format(best_step))
                        print()
                    else:
                        summary_writer.flush()
                        saver_best.save(sess, os.path.join(config['folder_out'], 'best'), global_step=step,
                                        latest_filename=config['ckpt_best'], write_meta_graph=False)
                        saver_curr.save(sess, os.path.join(config['folder_out'], 'curr'), global_step=step,
                                        latest_filename=config['ckpt_curr'], write_meta_graph=False)
                        with open(path_ckpt_step, 'wb') as f:
                            pickle.dump({'step': step, 'best_step': best_step, 'best_loss': best_loss}, f)
                        print('Step: {}/{}'.format(step, config['num_steps']))
    return


def get_test_ops(strategy, config, data_loaders, net, phase='test'):
    def get_replica(images, segments, overlaps):
        results = {}
        for idx_run in range(config['num_tests']):
            sub_results, _, _ = net(images, segments, overlaps, phase_param['num_slots'])
            for key, val in sub_results.items():
                if key in ['image']:
                    continue
                if key in results:
                    results[key].append(val)
                else:
                    results[key] = [val]
        results = {key: tf.stack(val, axis=0) for key, val in results.items()}
        return results
    with tf.name_scope('{}_ops'.format(phase)):
        with tf.name_scope('data'):
            data = data_loaders[phase].get_next()
        phase_param = config['phase_param'][phase]
        op_results = strategy.experimental_run_v2(get_replica, args=(data['image'], data['segment'], data['overlap']))
        with tf.name_scope('merge'):
            if strategy.num_replicas_in_sync > 1:
                op_results = {key: tf.concat(val.values, axis=1) for key, val in op_results.items()}
        op_results = {key: tf.cast(tf.clip_by_value(val, 0, 1) * 255, dtype=tf.uint8) for key, val in op_results.items()}
    return op_results


def test_model(strategy, config, data_loaders, net):
    def get_path_detail():
        return os.path.join(config['folder_out'], '{}.h5'.format(phase))
    phase_list = [n for n in config['phase_param'] if n not in ['train', 'valid']]
    run_dict = {}
    for phase in phase_list:
        path_detail = get_path_detail()
        if os.path.exists(path_detail):
            raise FileExistsError(path_detail)
        run_dict[phase] = get_test_ops(strategy, config, data_loaders, net, phase=phase)
    net_param = net.get_all_variables()
    saver_best = tf.train.Saver(net_param)
    with tf.Session() as sess:
        path_best_param = tf.train.latest_checkpoint(config['folder_out'], latest_filename=config['ckpt_best'])
        saver_best.restore(sess, path_best_param)
        for phase, results_op in run_dict.items():
            sess.run(data_loaders[phase].initializer)
            path_detail = get_path_detail()
            with h5py.File(path_detail, 'w') as f:
                while True:
                    try:
                        results = sess.run(results_op)
                        for key, val in results.items():
                            if key in f:
                                f[key].resize(f[key].shape[1] + val.shape[1], axis=1)
                                f[key][:, -val.shape[1]:] = val
                            else:
                                f.create_dataset(
                                    key, data=val, maxshape=[val.shape[0], None, *val.shape[2:]], compression='gzip')
                    except tf.errors.OutOfRangeError:
                        break
    return
