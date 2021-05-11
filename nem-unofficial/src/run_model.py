import gzip
import h5py
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from metric import compute_metrics


def compute_loss_coef(config, epoch):
    loss_coef = {'elbo': 1}
    for key, val in config['loss_coef'].items():
        epoch_list = [0] + val['epoch'] + [config['num_epochs'] - 1]
        assert len(epoch_list) == len(val['value'])
        assert len(epoch_list) == len(val['linear']) + 1
        assert epoch_list == sorted(epoch_list)
        for idx in range(len(epoch_list) - 1):
            if epoch <= epoch_list[idx + 1]:
                ratio = (epoch - epoch_list[idx]) / (epoch_list[idx + 1] - epoch_list[idx])
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


def compute_overview(config, images, results, dpi=150):
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
    summ_image_count = config['summ_image_count']
    image_batch = images[:summ_image_count].data.cpu().numpy()
    recon_batch = results['recon'][:summ_image_count].data.cpu().numpy()
    apc_batch = results['apc'][:summ_image_count].data.cpu().numpy()
    mask_batch = results['mask'][:summ_image_count].data.cpu().numpy()
    pres_batch = results['pres'][:summ_image_count].data.cpu().numpy()
    overview_list = [get_overview(idx) for idx in range(summ_image_count)]
    overview = np.concatenate(overview_list, axis=0)
    overview = np.moveaxis(overview, 2, 0)
    return overview


def train_model(config, data_loaders, net):
    optimizer = optim.Adam(net.parameters(), lr=config['lr'])
    phase_list = ['train']
    save_phase = 'train'
    if 'valid' in data_loaders:
        phase_list.append('valid')
        save_phase = 'valid'
    path_ckpt = os.path.join(config['folder_out'], config['file_ckpt'])
    path_model = os.path.join(config['folder_out'], config['file_model'])
    if config['resume']:
        checkpoint = torch.load(path_ckpt)
        start_epoch = checkpoint['epoch'] + 1
        best_epoch = checkpoint['best_epoch']
        best_loss = checkpoint['best_loss']
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Resume training from epoch {}'.format(start_epoch))
    else:
        start_epoch = 0
        best_epoch = -1
        best_loss = float('inf')
        print('Start training')
    print()
    with SummaryWriter(log_dir=config['folder_log'], purge_step=start_epoch) as writer:
        for epoch in range(start_epoch, config['num_epochs']):
            loss_coef = compute_loss_coef(config, epoch)
            print('Epoch: {}/{}'.format(epoch, config['num_epochs'] - 1))
            for phase in phase_list:
                phase_param = config['phase_param'][phase]
                net.train(phase == 'train')
                sum_losses, sum_metrics = {}, {}
                num_data = 0
                for idx_batch, data in enumerate(data_loaders[phase]):
                    batch_size = data['image'].shape[0]
                    if idx_batch == 0 and epoch % config['summ_image_intvl'] == 0:
                        with torch.set_grad_enabled(False):
                            results, _, _ = net(data, phase_param)
                        overview = compute_overview(config, data['image'][:, -1], results)
                        writer.add_image(phase.capitalize(), overview, global_step=epoch)
                        writer.flush()
                    with torch.set_grad_enabled(phase == 'train'):
                        results, metrics, losses = net(data, phase_param, require_extra=False)
                    for key, val in losses.items():
                        if key in sum_losses:
                            sum_losses[key] += val.sum().item()
                        else:
                            sum_losses[key] = val.sum().item()
                    for key, val in metrics.items():
                        if key in sum_metrics:
                            sum_metrics[key] += val.sum().item()
                        else:
                            sum_metrics[key] = val.sum().item()
                    num_data += batch_size
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss_opt = torch.stack(
                            [loss_coef[key] * val.mean() for key, val in losses.items() if key != 'compare']).sum()
                        loss_opt.backward()
                        optimizer.step()
                mean_losses = {key: val / num_data for key, val in sum_losses.items()}
                mean_metrics = {key: val / num_data for key, val in sum_metrics.items()}
                if epoch % config['summ_scalar_intvl'] == 0:
                    for key, val in mean_losses.items():
                        writer.add_scalar('{}/loss_{}'.format(phase.capitalize(), key), val, global_step=epoch)
                    for key, val in mean_metrics.items():
                        writer.add_scalar('{}/metric_{}'.format(phase.capitalize(), key), val, global_step=epoch)
                    writer.flush()
                print(phase.capitalize())
                print((' ' * 4).join([
                    'ARI_O: {:.3f}'.format(mean_metrics['ari_obj']),
                    'MSE: {:.2e}'.format(mean_metrics['mse']),
                    'LL: {:.1f}'.format(mean_metrics['ll']),
                    'Count: {:.3f}'.format(mean_metrics['count']),
                ]))
                if phase == save_phase:
                    if mean_losses['compare'] < best_loss:
                        best_loss = mean_losses['compare']
                        best_epoch = epoch
                        torch.save(net.state_dict(), path_model)
            save_dict = {
                'epoch': epoch,
                'best_epoch': best_epoch,
                'best_loss': best_loss,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(save_dict, path_ckpt)
            print('Best Epoch: {}'.format(best_epoch))
            print()
    return


def test_model(config, data_loaders, net):
    def get_path_detail():
        return os.path.join(config['folder_out'], '{}.h5'.format(phase))
    def get_path_metric():
        return os.path.join(config['folder_out'], '{}.pkl'.format(phase))
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
    for phase in phase_list:
        batch_size = config['batch_size']
        phase_param = config['phase_param'][phase]
        data_key = phase_param['key'] if 'key' in phase_param else phase
        path_detail = get_path_detail()
        metrics_all = {}
        with h5py.File(config['path_data'], 'r') as f_data, h5py.File(path_detail, 'r') as f_result:
            data = {key: f_data[data_key][key] for key in f_data[data_key]}
            results_all = {key: f_result[key] for key in f_result}
            for offset in range(0, data['image'].shape[0], batch_size):
                sub_data = {key: val[offset:offset + batch_size] for key, val in data.items()}
                sub_results_all = {key: val[:, offset:offset + batch_size] for key, val in results_all.items()}
                for key, val in sub_data.items():
                    if key in ['segment', 'overlap']:
                        sub_data[key] = val.astype(np.int)
                    else:
                        sub_data[key] = val.astype(np.float) / 255
                sub_results_all = {key: val.astype(np.float) / 255 for key, val in sub_results_all.items()}
                sub_metrics_all = compute_metrics(config, phase_param, sub_data, sub_results_all)
                for key, val in sub_metrics_all.items():
                    if key in metrics_all:
                        metrics_all[key].append(val)
                    else:
                        metrics_all[key] = [val]
        for key, val in metrics_all.items():
            if isinstance(val[0], tuple):
                metrics_all[key] = tuple([np.concatenate(n, axis=1) for n in zip(*val)])
            else:
                metrics_all[key] = np.concatenate(val, axis=1)
        path_metric = get_path_metric()
        with gzip.open(path_metric, 'wb') as f:
            pickle.dump(metrics_all, f)
    return
