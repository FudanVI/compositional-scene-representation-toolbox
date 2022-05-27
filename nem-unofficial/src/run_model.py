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
    loss_coef = {'elbo': 1}
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
    return loss_coef


def compute_overview(config, images, results, dpi=100):
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
        image = image_batch[fig_idx]
        recon = recon_batch[fig_idx]
        apc = apc_batch[fig_idx]
        mask = mask_batch[fig_idx]
        pres = pres_batch[fig_idx]
        rows, cols = 2, apc.shape[0] + 1
        fig, axes = plt.subplots(rows, cols, figsize=(cols, rows + 0.2), dpi=dpi)
        plot_image(axes[0, 0], convert_image(image), xlabel='scene')
        plot_image(axes[1, 0], convert_image(recon))
        for idx in range(apc.shape[0]):
            xlabel = 'slot_{}'.format(idx)
            color = [1.0, 0.5, 0.0] if pres[idx] >= 128 else [0.0, 0.5, 1.0]
            plot_image(axes[0, idx + 1], convert_image(apc[idx]), xlabel=xlabel, color=color)
            plot_image(axes[1, idx + 1], convert_image(mask[idx]))
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        out = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, -1)
        plt.close(fig)
        return out
    summ_image_count = min(config['summ_image_count'], config['batch_size'])
    image_batch = images[:summ_image_count].data.cpu().numpy()
    recon_batch = results['recon'][:summ_image_count].data.cpu().numpy()
    apc_batch = results['apc'][:summ_image_count].data.cpu().numpy()
    mask_batch = results['mask'][:summ_image_count].data.cpu().numpy()
    pres_batch = results['pres'][:summ_image_count].data.cpu().numpy()
    overview_list = [get_overview(idx) for idx in range(summ_image_count)]
    overview = np.concatenate(overview_list, axis=0)
    overview = np.moveaxis(overview, 2, 0)
    return overview


def add_scalars(writer, metrics, losses, step, phase):
    for key, val in metrics.items():
        writer.add_scalar('{}/metric_{}'.format(phase, key), val, global_step=step)
    for key, val in losses.items():
        writer.add_scalar('{}/loss_{}'.format(phase, key), val, global_step=step)
    return


def accumulate_values(sum_values, values):
    for key, val in values.items():
        if key in sum_values:
            sum_values[key] += val.sum().item()
        else:
            sum_values[key] = val.sum().item()
    return sum_values


def train_model(config, data_loaders, net):
    def data_loader_train():
        while True:
            for x in data_loaders['train']:
                yield x
    data_loader_valid = data_loaders['valid']
    phase_param_train = config['phase_param']['train']
    phase_param_valid = config['phase_param']['valid']
    optimizer = optim.Adam(net.parameters(), lr=config['lr'])
    path_ckpt = os.path.join(config['folder_out'], config['file_ckpt'])
    path_model = os.path.join(config['folder_out'], config['file_model'])
    if config['resume']:
        checkpoint = torch.load(path_ckpt)
        step = checkpoint['step']
        best_step = checkpoint['best_step']
        best_loss = checkpoint['best_loss']
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Resume training from step {}'.format(step))
    else:
        step = 0
        best_step = -1
        best_loss = float('inf')
        if config['path_pretrain'] is not None:
            net.load_state_dict(torch.load(config['path_pretrain']))
        print('Start training')
    print()
    with SummaryWriter(log_dir=config['folder_log'], purge_step=step + 1) as writer:
        for data_train in data_loader_train():
            step += 1
            if step > config['num_steps']:
                break
            loss_coef_train = compute_loss_coef(config, step)
            net.train(True)
            with torch.set_grad_enabled(True):
                _, metrics, losses = net(data_train, phase_param_train, require_extra=False)
            metrics = {key: val.mean() for key, val in metrics.items()}
            losses = {key: val.mean() for key, val in losses.items()}
            add_scalars(writer, metrics, losses, step, 'train')
            loss_opt = torch.stack(
                [loss_coef_train[key] * val for key, val in losses.items() if key != 'compare']).sum()
            optimizer.zero_grad()
            loss_opt.backward()
            optimizer.step()
            if step % config['ckpt_intvl'] == 0:
                with torch.set_grad_enabled(False):
                    results, _, _ = net(data_train, phase_param_train)
                overview = compute_overview(config, data_train['image'][:, -1], results)
                writer.add_image('train', overview, global_step=step)
                net.train(False)
                sum_metrics, sum_losses = {}, {}
                num_data = 0
                for idx_batch, data_valid in enumerate(data_loader_valid):
                    batch_size = data_valid['image'].shape[0]
                    with torch.set_grad_enabled(False):
                        results, metrics, losses = net(data_valid, phase_param_valid, require_extra=idx_batch == 0)
                    if idx_batch == 0:
                        overview = compute_overview(config, data_valid['image'][:, -1], results)
                        writer.add_image('valid', overview, global_step=step)
                    sum_metrics = accumulate_values(sum_metrics, metrics)
                    sum_losses = accumulate_values(sum_losses, losses)
                    num_data += batch_size
                mean_metrics = {key: val / num_data for key, val in sum_metrics.items()}
                mean_losses = {key: val / num_data for key, val in sum_losses.items()}
                loss_compare = mean_losses['compare']
                add_scalars(writer, mean_metrics, mean_losses, step, 'valid')
                writer.flush()
                if loss_compare < best_loss:
                    best_loss = loss_compare
                    best_step = step
                    torch.save(net.state_dict(), path_model)
                save_dict = {
                    'step': step,
                    'best_step': best_step,
                    'best_loss': best_loss,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(save_dict, path_ckpt)
                print('Step: {}/{}'.format(step, config['num_steps']))
                print((' ' * 4).join([
                    'ARI_O: {:.3f}'.format(mean_metrics['ari_obj']),
                    'MSE: {:.2e}'.format(mean_metrics['mse']),
                    'LL: {:.1f}'.format(mean_metrics['ll']),
                    'Count: {:.3f}'.format(mean_metrics['count']),
                ]))
                print('Best Step: {}'.format(best_step))
                print()
    return


def test_model(config, data_loaders, net):
    def get_path_detail():
        return os.path.join(config['folder_out'], '{}.h5'.format(phase))
    phase_list = [n for n in config['phase_param'] if n not in ['train', 'valid']]
    for phase in phase_list:
        path_detail = get_path_detail()
        if os.path.exists(path_detail):
            raise FileExistsError(path_detail)
    path_model = os.path.join(config['folder_out'], config['file_model'])
    net.load_state_dict(torch.load(path_model))
    net.train(False)
    for phase in phase_list:
        phase_param = config['phase_param'][phase]
        path_detail = get_path_detail()
        data_key = phase_param['key'] if 'key' in phase_param else phase
        results_all = {}
        for data in data_loaders[data_key]:
            results = {}
            for idx_run in range(config['num_tests']):
                with torch.set_grad_enabled(False):
                    sub_results, _, _ = net(data, phase_param)
                for key, val in sub_results.items():
                    val = val.data.cpu().numpy()
                    if key != 'pres':
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
