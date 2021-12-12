import h5py
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


def compute_loss_coef(config, step=None):
    step_last = config['num_steps']
    if step is None:
        step = step_last
    loss_coef = {'nll': 1, 'kld_pres': 1, 'kld_view': 1, 'kld_attr_obj': 1, 'kld_attr_bck': 1}
    for key, val in config['loss_coef'].items():
        step_list = [1] + val['step'] + [step_last]
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
    for name in ['kld', 'reg']:
        if 'sched_{}'.format(name) in loss_coef:
            coef = loss_coef['sched_{}'.format(name)]
            for key in loss_coef:
                if key.split('_')[0] == name:
                    loss_coef[key] *= coef
    return loss_coef


def compute_overview(config, outputs, dpi=100):
    def convert_image(image):
        image = np.moveaxis(image, 0, 2)
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
        image = results_sel['image'][fig_idx]
        recon = results_sel['recon'][fig_idx]
        noisy_recon = results_sel['noisy_recon'][fig_idx]
        recon_soft = results_sel['recon_soft'][fig_idx]
        apc = results_sel['apc'][fig_idx]
        noisy_apc = results_sel['noisy_apc'][fig_idx]
        shp = results_sel['shp'][fig_idx]
        shp_soft = results_sel['shp_soft'][fig_idx]
        mask = results_sel['mask'][fig_idx]
        mask_soft = results_sel['mask_soft'][fig_idx]
        pres = results_sel['pres'][fig_idx]
        num_views, num_slots = apc.shape[:2]
        rows, cols = 6 * num_views, num_slots + 1
        fig, axes = plt.subplots(rows, cols, figsize=(cols, rows + 0.2), dpi=dpi)
        for idx_v in range(num_views):
            plot_image(axes[idx_v * 6, 0], convert_image(image[idx_v]), xlabel='scene' if idx_v == 0 else None)
            plot_image(axes[idx_v * 6 + 1, 0], convert_image(noisy_recon[idx_v]))
            plot_image(axes[idx_v * 6 + 2, 0], convert_image(recon[idx_v]))
            plot_image(axes[idx_v * 6 + 3, 0], convert_image(recon_soft[idx_v]))
            axes[idx_v * 6 + 4, 0].set_visible(False)
            axes[idx_v * 6 + 5, 0].set_visible(False)
            for idx_s in range(num_slots):
                xlabel = 'obj_{}'.format(idx_s) if idx_s < num_slots - 1 else 'back'
                xlabel = xlabel if idx_v == 0 else None
                color = [1.0, 0.5, 0.0] if pres[idx_s] >= 128 else [0.0, 0.5, 1.0]
                plot_image(axes[idx_v * 6, idx_s + 1], convert_image(apc[idx_v, idx_s]), xlabel=xlabel, color=color)
                plot_image(axes[idx_v * 6 + 1, idx_s + 1], convert_image(noisy_apc[idx_v, idx_s]))
                plot_image(axes[idx_v * 6 + 2, idx_s + 1], convert_image(shp[idx_v, idx_s]))
                plot_image(axes[idx_v * 6 + 3, idx_s + 1], convert_image(shp_soft[idx_v, idx_s]))
                plot_image(axes[idx_v * 6 + 4, idx_s + 1], convert_image(mask[idx_v, idx_s]))
                plot_image(axes[idx_v * 6 + 5, idx_s + 1], convert_image(mask_soft[idx_v, idx_s]))
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        out = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, -1)
        plt.close(fig)
        return out
    summ_image_count = min(config['summ_image_count'], config['batch_size'])
    results_sel = {key: val[:summ_image_count].data.cpu().numpy() for key, val in outputs['result'].items()}
    overview_list = [get_overview(idx) for idx in range(summ_image_count)]
    overview = np.concatenate(overview_list, axis=0)
    overview = np.moveaxis(overview, 2, 0)
    return overview


def convert_metrics_losses(outputs, loss_coef):
    metrics = {key: val.mean() for key, val in outputs['metric'].items()}
    losses = {key: val.mean() for key, val in outputs['loss'].items()}
    loss_opt = torch.stack([loss_coef[key] * val for key, val in losses.items()]).sum()
    return metrics, losses, loss_opt


def add_scalars(writer, metrics, losses, step, phase):
    for key, val in metrics.items():
        writer.add_scalar('{}/metric_{}'.format(phase, key), val, global_step=step)
    for key, val in losses.items():
        writer.add_scalar('{}/loss_{}'.format(phase, key), val, global_step=step)
    return

def add_overviews(config, writer, outputs, outputs_ext, outputs_view, outputs_attr, step, phase):
    overview = compute_overview(config, outputs)
    writer.add_image(phase, overview, global_step=step)
    overview_view = compute_overview(config, outputs_view)
    writer.add_image('{}_view'.format(phase), overview_view, global_step=step)
    overview_attr = compute_overview(config, outputs_attr)
    writer.add_image('{}_attr'.format(phase), overview_attr, global_step=step)
    return


def accumulate_values(sum_values, values):
    for key, val in values.items():
        if key in sum_values:
            sum_values[key] += val.sum().item()
        else:
            sum_values[key] = val.sum().item()
    return sum_values


def print_metrics(mean_metrics):
    print((' ' * 4).join([
        'ARI_A_S: {:.3f}'.format(mean_metrics['ari_soft_all_s']),
        'ARI_A_M: {:.3f}'.format(mean_metrics['ari_soft_all_m']),
        'ARI_O_S: {:.3f}'.format(mean_metrics['ari_soft_obj_s']),
        'ARI_O_M: {:.3f}'.format(mean_metrics['ari_soft_obj_m']),
    ]))
    print((' ' * 4).join([
        'MSE: {:.2e}'.format(mean_metrics['mse_soft']),
        'Count: {:.3f}'.format(mean_metrics['count']),
    ]))
    return


def train_model(config, data_loaders, net):
    def lr_lambda(x):
        decay_rate = config['lr_decay']
        decay_steps = config['decay_steps']
        warmup_steps = config['warmup_steps']
        decay_ratio = 0 if decay_steps == 0 else x / decay_steps
        decay_coef = pow(decay_rate, decay_ratio)
        warmup_ratio = 1 if warmup_steps == 0 else x / warmup_steps
        warmup_coef = min(warmup_ratio, 1)
        coef = decay_coef * warmup_coef
        return coef
    def data_loader_train():
        while True:
            for x in data_loaders['train']:
                yield x
    data_loader_valid = data_loaders['valid']
    phase_param_train = config['phase_param']['train']
    phase_param_valid = config['phase_param']['valid']
    optimizer = optim.Adam(net.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    path_ckpt = os.path.join(config['folder_out'], config['file_ckpt'])
    path_model = os.path.join(config['folder_out'], config['file_model'])
    if config['resume']:
        checkpoint = torch.load(path_ckpt)
        step = checkpoint['step']
        best_step = checkpoint['best_step']
        best_loss = checkpoint['best_loss']
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print('Resume training from step {}'.format(step))
    else:
        step = 0
        best_step = -1
        best_loss = float('inf')
        if config['path_pretrain'] is not None:
            state_dict = torch.load(config['path_pretrain'])
            if isinstance(state_dict, dict):
                state_dict = state_dict['model_state_dict']
            net.load_state_dict(state_dict)
        print('Start training')
    print()
    with SummaryWriter(log_dir=config['folder_log'], purge_step=step + 1) as writer:
        loss_coef_valid = compute_loss_coef(config)
        for data_train in data_loader_train():
            step += 1
            if step > config['num_steps']:
                break
            loss_coef_train = compute_loss_coef(config, step)
            single_view = step <= config['single_steps']
            net.train(True)
            with torch.set_grad_enabled(True):
                outputs, outputs_ext, _, _ = net(
                    data_train, phase_param_train, loss_coef_train, single_view=single_view,
                    require_results=False, require_results_gen=False)
            scalar_metrics, scalar_losses, loss_opt = convert_metrics_losses(outputs, loss_coef_train)
            add_scalars(writer, scalar_metrics, scalar_losses, step, 'train')
            optimizer.zero_grad()
            loss_opt.backward()
            optimizer.step()
            scheduler.step()
            if step % config['ckpt_intvl'] == 0:
                with torch.set_grad_enabled(False):
                    outputs, outputs_ext, outputs_view, outputs_attr = net(
                        data_train, phase_param_train, loss_coef_train, single_view=single_view)
                add_overviews(
                    config, writer, outputs, outputs_ext, outputs_view, outputs_attr, step, 'train')
                net.train(False)
                sum_metrics, sum_losses, sum_metrics_ext, sum_losses_ext = {}, {}, {}, {}
                num_data = 0
                for idx_batch, data_valid in enumerate(data_loader_valid):
                    require_results = idx_batch == 0
                    batch_size = data_valid['image'].shape[0]
                    with torch.set_grad_enabled(False):
                        outputs, outputs_ext, outputs_view, output_attr = net(
                            data_valid, phase_param_valid, loss_coef_valid, single_view=single_view,
                            require_results=require_results, require_results_gen=require_results)
                    if require_results:
                        add_overviews(
                            config, writer, outputs, outputs_ext, outputs_view, outputs_attr, step, 'valid')
                    outputs['loss']['compare'] = torch.stack(
                        [loss_coef_valid[key] * val for key, val in outputs['loss'].items()]).sum(0)
                    sum_metrics = accumulate_values(sum_metrics, outputs['metric'])
                    sum_losses = accumulate_values(sum_losses, outputs['loss'])
                    num_data += batch_size
                mean_metrics = {key: val / num_data for key, val in sum_metrics.items()}
                mean_losses = {key: val / num_data for key, val in sum_losses.items()}
                loss_compare = mean_losses['compare']
                add_scalars(writer, mean_metrics, mean_losses, step, 'valid')
                writer.flush()
                if loss_compare < best_loss and not single_view:
                    best_loss = loss_compare
                    best_step = step
                    torch.save(net.state_dict(), path_model)
                save_dict = {
                    'step': step,
                    'best_step': best_step,
                    'best_loss': best_loss,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }
                torch.save(save_dict, path_ckpt)
                idx_save = step // config['ckpt_intvl']
                if idx_save % config['save_intvl'] == 0:
                    name_save = 'save_{}.pth'.format(idx_save // config['save_intvl'])
                    path_save = os.path.join(config['folder_out'], name_save)
                    torch.save(save_dict, path_save)
                print('Step: {}/{}'.format(step, config['num_steps']))
                print_metrics(mean_metrics)
                print('Best Step: {}'.format(best_step))
                print()
    return


def test_model(config, data_loaders, net):
    def get_path_detail():
        return os.path.join(config['folder_out'], '{}.h5'.format(phase))
    phase_list = [n for n in config['phase_param'] if n not in ['train', 'valid']]
    path_model = os.path.join(config['folder_out'], config['file_model'])
    net.load_state_dict(torch.load(path_model))
    net.train(False)
    loss_coef = compute_loss_coef(config)
    for phase in phase_list:
        phase_param = config['phase_param'][phase]
        path_detail = get_path_detail()
        data_key = phase_param['key'] if 'key' in phase_param else phase
        results_all = {}
        for data in data_loaders[data_key]:
            results = {}
            for idx_run in range(config['num_tests']):
                with torch.set_grad_enabled(False):
                    outputs, outputs_ext, outputs_view, outputs_attr = net(
                        data, phase_param, loss_coef, single_view=False, infer_extra=False, require_results=True,
                        require_results_gen=False, deterministic_data=True)
                    sub_results = outputs['result']
                for key, val in sub_results.items():
                    if key in ['image', 'noisy_recon', 'noisy_apc', 'view_latent', 'attr_obj_latent', 'attr_bck_latent',
                               'logits_pres', 'zeta', 'logits_zeta', 'trs']:
                        continue
                    val = val.data.cpu().numpy()
                    if key in ['recon', 'recon_soft', 'mask', 'mask_soft', 'apc', 'shp', 'shp_soft']:
                        val = np.moveaxis(val, -3, -1)
                    if key in results:
                        results[key].append(val)
                    else:
                        results[key] = [val]
            for key, val in results.items():
                val = np.stack(val)
                if key in results_all:
                    results_all[key].append(val)
                else:
                    results_all[key] = [val]
        with h5py.File(path_detail, 'w') as f:
            for key, val in results_all.items():
                f.create_dataset(key, data=np.concatenate(val, axis=1), compression='gzip')
    return


def test_model_multi(config, data_loaders, net):
    def get_path_detail():
        return os.path.join(config['folder_out'], '{}_{}.h5'.format(phase, num_views))
    phase_list = [n for n in config['phase_param'] if n not in ['train', 'valid']]
    path_model = os.path.join(config['folder_out'], config['file_model'])
    net.load_state_dict(torch.load(path_model))
    net.train(False)
    loss_coef = compute_loss_coef(config)
    for phase in phase_list:
        phase_param = config['phase_param'][phase]
        data_key = phase_param['key'] if 'key' in phase_param else phase
        for num_views in [1, 2, 4, 8]:
            phase_param['num_views'] = num_views
            path_detail = get_path_detail()
            results_all = {}
            for data in data_loaders[data_key]:
                results = {}
                for idx_run in range(config['num_tests']):
                    with torch.set_grad_enabled(False):
                        outputs, outputs_ext, outputs_view, outputs_attr = net(
                            data, phase_param, loss_coef, single_view=False, infer_extra=False, require_results=True,
                            require_results_gen=False, deterministic_data=True)
                        sub_results = outputs['result']
                    for key, val in sub_results.items():
                        if key in ['image', 'noisy_recon', 'noisy_apc', 'view_latent', 'attr_obj_latent', 'attr_bck_latent',
                                   'logits_pres', 'zeta', 'logits_zeta', 'trs']:
                            continue
                        val = val.data.cpu().numpy()
                        if key in ['recon', 'recon_soft', 'mask', 'mask_soft', 'apc', 'shp', 'shp_soft']:
                            val = np.moveaxis(val, -3, -1)
                        if key in results:
                            results[key].append(val)
                        else:
                            results[key] = [val]
                for key, val in results.items():
                    val = np.stack(val)
                    if key in results_all:
                        results_all[key].append(val)
                    else:
                        results_all[key] = [val]
            with h5py.File(path_detail, 'w') as f:
                for key, val in results_all.items():
                    f.create_dataset(key, data=np.concatenate(val, axis=1), compression='gzip')
    return


def test_model_cond(config, data_loaders, net):
    def get_path_detail():
        return os.path.join(config['folder_out'], 'cond_{}_{}.h5'.format(phase, num_views))
    phase_list = [n for n in config['phase_param'] if n not in ['train', 'valid']]
    path_model = os.path.join(config['folder_out'], config['file_model'])
    net.load_state_dict(torch.load(path_model))
    net.train(False)
    loss_coef = compute_loss_coef(config)
    for phase in phase_list:
        phase_param = config['phase_param'][phase]
        data_key = phase_param['key'] if 'key' in phase_param else phase
        for num_views in [1, 2, 4]:
            phase_param['num_views'] = num_views
            path_detail = get_path_detail()
            results_all = {}
            for data in data_loaders[data_key]:
                results = {}
                for idx_run in range(config['num_tests']):
                    with torch.set_grad_enabled(False):
                        outputs, outputs_ext, outputs_view, outputs_attr = net(
                            data, phase_param, loss_coef, single_view=False, infer_extra=True, require_results=True,
                            require_results_gen=False, deterministic_data=True)
                        sub_results = outputs_ext['result_2']
                    for key, val in sub_results.items():
                        if key in ['image', 'noisy_recon', 'noisy_apc', 'view_latent', 'attr_obj_latent', 'attr_bck_latent',
                                   'logits_pres', 'zeta', 'logits_zeta', 'trs']:
                            continue
                        val = val.data.cpu().numpy()
                        if key in ['recon', 'recon_soft', 'mask', 'mask_soft', 'apc', 'shp', 'shp_soft']:
                            val = np.moveaxis(val, -3, -1)
                        if key in results:
                            results[key].append(val)
                        else:
                            results[key] = [val]
                for key, val in results.items():
                    val = np.stack(val)
                    if key in results_all:
                        results_all[key].append(val)
                    else:
                        results_all[key] = [val]
            with h5py.File(path_detail, 'w') as f:
                for key, val in results_all.items():
                    f.create_dataset(key, data=np.concatenate(val, axis=1), compression='gzip')
    return
