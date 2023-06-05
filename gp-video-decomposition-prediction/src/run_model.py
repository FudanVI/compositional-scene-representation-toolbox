import math
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


def add_scalars(writer, metrics, losses, step, phase, stage):
    for mode in ['T', 'Q']:
        state = 'observe' if mode == 'T' else 'predict'
        if mode in metrics:
            for key, val in metrics[mode].items():
                writer.add_scalar('stage_{}/{}/{}_metric_{}'.format(stage, phase, state, key), val, global_step=step)
        if mode in losses:
            for key, val in losses[mode].items():
                writer.add_scalar('stage_{}/{}/{}_loss_{}'.format(stage, phase, state, key), val, global_step=step)
    return


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

    def get_overview(fig_idx, mode):
        image = results_sel[mode]['image'][fig_idx]
        recon = results_sel[mode]['recon'][fig_idx]
        apc = results_sel[mode]['apc'][fig_idx]
        noisy_recon = results_sel[mode]['noisy_recon'][fig_idx]
        noisy_apc = results_sel[mode]['noisy_apc'][fig_idx]
        shp = results_sel[mode]['shp'][fig_idx]
        mask = results_sel[mode]['mask'][fig_idx]
        pres = results_sel[mode]['pres'][fig_idx]
        num_views, num_slots = apc.shape[:2]
        rows, cols = 4 * num_views, num_slots + 1
        fig, axes = plt.subplots(rows, cols, figsize=(cols, rows + 0.2), dpi=dpi)
        for idx_v in range(num_views):
            plot_image(axes[idx_v * 4, 0], convert_image(image[idx_v]), xlabel='scene' if idx_v == 0 else None,
                       ylabel='shp')
            plot_image(axes[idx_v * 4 + 1, 0], convert_image(noisy_recon[idx_v]), ylabel='noisy apc')
            plot_image(axes[idx_v * 4 + 2, 0], convert_image(recon[idx_v]), ylabel='apc')
            plot_image(axes[idx_v * 4 + 3, 0], convert_image(recon[idx_v]), ylabel='mask')
            for idx_s in range(num_slots):
                xlabel = 'obj_{}'.format(idx_s) if idx_s < num_slots - 1 else 'back'
                xlabel = xlabel if idx_v == 0 else None
                color = [1.0, 0.5, 0.0] if pres[idx_s] >= 128 else [0.0, 0.5, 1.0]
                plot_image(axes[idx_v * 4, idx_s + 1], convert_image(shp[idx_v, idx_s]), xlabel=xlabel, color=color)
                plot_image(axes[idx_v * 4 + 1, idx_s + 1], convert_image(noisy_apc[idx_v, idx_s]))
                plot_image(axes[idx_v * 4 + 2, idx_s + 1], convert_image(apc[idx_v, idx_s]))
                plot_image(axes[idx_v * 4 + 3, idx_s + 1], convert_image(mask[idx_v, idx_s]))
        fig.tight_layout(pad=0)
        if mode == 'Q':
            fig.set_facecolor('aquamarine')
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        out = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, -1)
        plt.close(fig)
        return out

    summ_image_count = min(config['summ_image_count'], config['batch_size'])
    results_sel = {}
    for mode in outputs['result'].keys():
        results_sel[mode] = {key: val[:summ_image_count].data.cpu().numpy() for key, val in
                             outputs['result'][mode].items()}
    overview_list = []
    keys = outputs['result'].keys()
    for idx in range(summ_image_count):
        for mode in keys:
            overview_list.append(get_overview(idx, mode))
    overview = np.concatenate(overview_list, axis=0)
    overview = np.moveaxis(overview, 2, 0)
    return overview


def add_overviews(config, writer, outputs, step, phase, stage):
    overview = compute_overview(config, outputs)
    writer.add_image('stage_{}/{}'.format(stage, phase), overview, global_step=step)


def accumulate_values(sum_values, values):
    for mode in values:
        for key, val in values[mode].items():
            if key in sum_values[mode]:
                sum_values[mode][key] += val.sum().item()
            else:
                sum_values[mode][key] = val.sum().item()
    return sum_values


def compute_view_ceof(config, step=None):
    step_last = config['num_steps']
    observe_views = 0
    step_list = config['view_coef']['step'] + [step_last]
    for idx in range(len(step_list)):
        if step <= step_list[idx]:
            observe_views = config['view_coef']['value'][idx]
            break
    return observe_views


def compute_loss_coef(config, step=None):
    step_last = config['num_steps']
    if step is None:
        step = step_last
    loss_coef = {'T': {'nll': 1, 'kld_pres': 1, 'kld_view': 1, 'kld_lam': 1, 'kld_attr_obj': 1, 'kld_attr_bck': 1},
                 'Q': {'nll': 1, 'kld_view': 1, 'kld_lam': 1}}
    for key, val in config['loss_coef'].items():
        if key in ['T', 'Q']:
            continue
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
    for mode in ['T', 'Q']:
        if mode in config['loss_coef']:
            key_list = list(config['loss_coef'][mode].keys())
            for key in key_list:
                val = config['loss_coef'][mode][key]
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
                            loss_coef[mode][key] = (1 - ratio) * val_1 + ratio * val_2
                        else:
                            loss_coef[mode][key] = math.exp((1 - ratio) * math.log(val_1) + ratio * math.log(val_2))
                        assert math.isfinite(loss_coef[mode][key])
                        break
                else:
                    raise ValueError
    for mode in ['T', 'Q']:
        if mode not in loss_coef:
            continue
        for name in ['kld', 'reg']:
            if 'sched_{}'.format(name) in loss_coef[mode]:
                coef = loss_coef[mode]['sched_{}'.format(name)]
                for key in loss_coef[mode]:
                    if key.split('_')[0] == name:
                        loss_coef[mode][key] *= coef
    return loss_coef


def convert_metrics_losses(outputs, loss_coef):
    metrics = {}
    losses = {}
    for mode in ['T', 'Q']:
        if mode in outputs['metric']:
            metrics[mode] = {key: val.mean() for key, val in outputs['metric'][mode].items()}
        if mode in outputs['loss']:
            losses[mode] = {key: val.mean() for key, val in outputs['loss'][mode].items()}
    loss_opt = 0
    for mode in ['T', 'Q']:
        if mode in losses:
            loss_opt += torch.stack([loss_coef[mode][key] * val for key, val in losses[mode].items()]).sum()
    return metrics, losses, loss_opt


def print_metrics(mean_metrics):
    print((' ' * 4).join([
        'ARI_A_S: {:.3f}'.format(mean_metrics['ari_all_s']),
        'ARI_A_M: {:.3f}'.format(mean_metrics['ari_all_m']),
        'ARI_O_S: {:.3f}'.format(mean_metrics['ari_obj_s']),
        'ARI_O_M: {:.3f}'.format(mean_metrics['ari_obj_m']),
    ]))
    print((' ' * 4).join([
        'MSE: {:.2e}'.format(mean_metrics['mse']),
        'Count: {:.3f}'.format(mean_metrics['count']),
    ]))
    return


def train_model(config, data_loaders, model):
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

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    data_loader_valid = data_loaders['valid']
    phase_param_train = config['phase_param']['train']
    phase_param_valid = config['phase_param']['valid']
    path_ckpt = os.path.join(config['folder_out'], config['stage'], config['file_ckpt'])
    path_model = os.path.join(config['folder_out'], config['stage'], config['file_model'])
    if config['stage'] == 'two':
        for view in config['view_coef']['value']:
            path = os.path.join(config['folder_out'], config['stage'], 'view_{}'.format(view))
            if not os.path.exists(path):
                os.makedirs(path)
    if config['resume']:
        checkpoint = torch.load(path_ckpt)
        step = checkpoint['step']
        best_step = checkpoint['best_step']
        best_loss = checkpoint['best_loss']
        for val in config['view_coef']['value']:
            if val not in best_loss:
                best_loss.update({val: float('inf')})
            if val not in best_step:
                best_step.update({val: -1})
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        del checkpoint
        print('Resume training from step {} of stage {}'.format(step, config['stage']))
    else:
        if config['stage'] == 'two' and config['pretrain']:
            print('loading pretrain...')
            pretrain = os.path.join(config['folder_out'], 'one', config['file_model'])
            checkpoint = torch.load(pretrain)
            model.load_state_dict(checkpoint, strict=False)
        step = 0
        if config['stage'] == 'two':
            best_loss = {view: float('inf') for view in config['view_coef']['value']}
            best_step = {view: -1 for view in config['view_coef']['value']}
        else:
            best_loss = float('inf')
            best_step = -1
        print('Start training stage {}'.format(config['stage']))
    print()

    with SummaryWriter(log_dir=config['folder_log'], purge_step=step + 1) as writer:
        loss_coef_valid = compute_loss_coef(config)
        for data_train in data_loader_train():
            step += 1
            if step > config['num_steps']:
                break
            loss_coef_train = compute_loss_coef(config, step)
            single_view = step <= config['single_steps']
            if config['stage'] == 'two' and not single_view:
                total_views = compute_view_ceof(config, step)
                query_num = np.random.randint(config['min_query_num'], config['max_query_num'] + 1)
                phase_param_train['observed_views'] = total_views - query_num
                phase_param_train['num_views'] = total_views
                phase_param_valid['observed_views'] = total_views - query_num
                phase_param_valid['num_views'] = total_views
            model.train(True)
            # with torch.set_grad_enabled(True):
            with torch.set_grad_enabled(True):
                outputs = model(
                    data_train, phase_param_train, loss_coef_train, single_view=single_view,
                    require_results=False, stage=config['stage'], determine_data=config['determine_data'],
                    continuous=True)
                scalar_metrics, scalar_losses, loss_opt = convert_metrics_losses(outputs, loss_coef_train)
                add_scalars(writer, scalar_metrics, scalar_losses, step, 'train', config['stage'])
                optimizer.zero_grad()
                loss_opt.backward()
                optimizer.step()
                scheduler.step()
                if step % config['ckpt_intvl'] == 0:
                    with torch.set_grad_enabled(False):
                        outputs = model(data_train, phase_param_train, loss_coef_train, single_view=single_view,
                                        require_results=True, stage=config['stage'],
                                        determine_data=config['determine_data'])
                    add_overviews(config, writer, outputs, step, 'train', config['stage'])
                    model.train(False)
                    sum_metrics, sum_losses, sum_metrics_ext, sum_losses_ext = {'T': {}}, {'T': {}}, {
                        'T': {}}, {'T': {}}
                    add = {'Q': {}}
                    if config['stage'] == 'two' and not single_view:
                        sum_metrics.update(add)
                        sum_losses.update(add)
                        sum_metrics_ext.update(add)
                        sum_losses_ext.update(add)
                    num_data = 0
                    sum_compare_loss = 0
                    for idx_batch, data_valid in enumerate(data_loader_valid):
                        require_results = idx_batch == 0
                        batch_size = data_valid['image'].shape[0]
                        with torch.set_grad_enabled(False):
                            outputs = model(
                                data_valid, phase_param_valid, loss_coef_valid, single_view=single_view,
                                require_results=require_results, stage=config['stage'],
                                determine_data=config['determine_data'])
                        num_data += batch_size
                        if require_results:
                            add_overviews(config, writer, outputs, step, 'valid', config['stage'])
                        loss_valid = 0
                        for mode in ['T', 'Q']:
                            if mode in outputs['loss']:
                                loss_valid += torch.stack(
                                    [loss_coef_valid[mode][key] * val for key, val in
                                     outputs['loss'][mode].items()]).sum()
                        sum_compare_loss += loss_valid.item()
                        sum_metrics = accumulate_values(sum_metrics, outputs['metric'])
                        sum_losses = accumulate_values(sum_losses, outputs['loss'])
                    mean_metrics = {mode: {key: val / num_data for key, val in sum_metrics[mode].items()} for mode in
                                    sum_metrics}
                    mean_losses = {mode: {key: val / num_data for key, val in sum_losses[mode].items()} for mode in
                                   sum_losses}
                    loss_compare = sum_compare_loss / num_data
                    if not config['mean_nll']:
                        loss_compare = loss_compare / phase_param_valid['num_views']
                    writer.add_scalar('stage_{}/{}/loss_compare'.format(config['stage'], 'valid'), loss_compare,
                                      global_step=step)
                    add_scalars(writer, mean_metrics, mean_losses, step, 'valid', config['stage'])
                    writer.flush()

                    if not single_view:
                        if config['stage'] == 'one' and loss_compare < best_loss:
                            best_loss = loss_compare
                            best_step = step
                            torch.save(model.state_dict(), path_model)
                        if config['stage'] == 'two' and loss_compare < best_loss[total_views]:
                            best_loss[total_views] = loss_compare
                            best_step[total_views] = step
                            path_model = os.path.join(config['folder_out'], config['stage'],
                                                      'view_{}'.format(total_views),
                                                      config['file_model'])
                            torch.save(model.state_dict(), path_model)

                    save_dict = {
                        'step': step,
                        'best_step': best_step,
                        'best_loss': best_loss,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }
                    torch.save(save_dict, path_ckpt)
                    idx_save = step // config['ckpt_intvl']
                    if idx_save % config['save_intvl'] == 0:
                        name_save = 'save_{}.pth'.format(idx_save // config['save_intvl'])
                        path_save = os.path.join(config['folder_out'], config['stage'], name_save)
                        torch.save(save_dict, path_save)
                    print('Step: {}/{}'.format(step, config['num_steps']))
                    for mode, val in mean_metrics.items():
                        if mode == 'T':
                            print('observed')
                        else:
                            print('predict')
                        print_metrics(val)
                    print('Best Step: {}'.format(best_step))
                    print()


def test_model_observe(config, data_loaders, model):
    def get_path_detail():
        return os.path.join(config['folder_out'], config['stage'], '{}_{}.h5'.format(phase, num_views))

    phase_list = [n for n in config['phase_param'] if n not in ['train', 'valid']]
    num_views_list = [1, 2, 4, 8]
    for phase in phase_list:
        for num_views in num_views_list:
            path_detail = get_path_detail()
            if os.path.exists(path_detail):
                os.remove(path_detail)
    path_model = os.path.join(config['folder_out'], config['stage'], config['file_model'])
    checkpoint = torch.load(path_model)
    model.load_state_dict(checkpoint, strict=False)
    model.train(False)
    loss_coef = compute_loss_coef(config)
    for phase in phase_list:
        phase_param = config['phase_param'][phase]
        for num_views in num_views_list:
            print('processing {} data: {} views'.format(phase, num_views))
            phase_param['num_views'] = num_views
            phase_param['observed_views'] = num_views
            path_detail = get_path_detail()
            results_all = {}
            for data in data_loaders[phase]:
                results = {}
                for idx_run in range(config['num_tests']):
                    outputs = model(data, phase_param, loss_coef, single_view=False, test_data=True,
                                    require_results=True, stage=config['stage'],
                                    determine_data=config['determine_data'])
                    sub_results = outputs['result']['T']
                    for key, val in sub_results.items():
                        if key not in ['image', 'recon', 'mask', 'apc', 'shp', 'pres', 'log_ord']:
                            continue
                        val = val.data.cpu().numpy()
                        if key in ['image', 'recon', 'mask', 'apc', 'shp']:
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


def test_model_predict(config, data_loaders, model):
    def get_path_detail():
        return os.path.join(config['folder_out'], config['stage'], 'predict',
                            '{}_m{}_o{}_q{}.h5'.format(phase, mode, observed_views, query_views))

    if not os.path.exists(os.path.join(config['folder_out'], config['stage'], 'predict')):
        os.makedirs(os.path.join(config['folder_out'], config['stage'], 'predict'))
    phase_list = [n for n in config['phase_param'] if n not in ['train', 'valid']]
    num_views = config['test_views']
    query_views_list = [n for n in range(1, 5)]
    observe_views_list = [num_views - n for n in query_views_list]
    for phase in phase_list:
        for mode in [1, 2, 3]:
            for observed_views, query_views in zip(observe_views_list, query_views_list):
                path_detail = get_path_detail()
                if os.path.exists(path_detail):
                    os.remove(path_detail)
    path_model = os.path.join(config['folder_out'], config['stage'], 'view_10', config['file_model'])
    checkpoint = torch.load(path_model)
    model.load_state_dict(checkpoint, strict=False)
    model.train(False)
    loss_coef = compute_loss_coef(config)
    for phase in phase_list:
        for mode in [1, 2, 3]:
            for observed_views, query_views in zip(observe_views_list, query_views_list):
                if mode == 1:
                    K = query_views + 1
                    query_views_index = [math.floor(i / K * num_views) - 1 for i in range(1, K)]
                    observe_views_index = [i for i in range(num_views) if i not in query_views_index]
                else:
                    observe_views_index = [*range(num_views)][:observed_views]
                    query_views_index = [*range(num_views)][observed_views:]
                if mode in [1, 2]:
                    kwargs = {'t_index': observe_views_index, 'q_index': query_views_index}
                else:
                    kwargs = {}
                phase_param = config['phase_param'][phase]
                print('processing data: {}, mode: {}, observe: {}, query: {}'.format(phase, mode, observed_views,
                                                                                     query_views))
                phase_param['num_views'] = num_views
                phase_param['observed_views'] = observed_views
                path_detail = get_path_detail()
                results_all = {'T': {}, 'Q': {}}
                for data in data_loaders[phase]:
                    results = {'T': {}, 'Q': {}}
                    for idx_run in range(config['num_tests']):
                        outputs = model(data, phase_param, loss_coef, single_view=False, test_data=True,
                                        require_results=True, stage=config['stage'],
                                        determine_data=True, **kwargs)
                        for choose in ['T', 'Q']:
                            sub_results = outputs['result'][choose]
                            for key, val in sub_results.items():
                                if key not in ['image', 'recon', 'mask', 'apc', 'shp', 'pres', 'log_ord', 'index']:
                                    continue
                                val = val.data.cpu().numpy()
                                if key in ['image', 'recon', 'mask', 'apc', 'shp']:
                                    val = np.moveaxis(val, -3, -1)
                                if key in results[choose]:
                                    results[choose][key].append(val)
                                else:
                                    results[choose][key] = [val]
                    for choose in ['T', 'Q']:
                        for key, val in results[choose].items():
                            val = np.stack(val)
                            if key in results_all[choose]:
                                results_all[choose][key].append(val)
                            else:
                                results_all[choose][key] = [val]
                with h5py.File(path_detail, 'w') as f:
                    for choose in results_all.keys():
                        f.create_group(choose)
                        for key, val in results_all[choose].items():
                            f[choose].create_dataset(key, data=np.concatenate(val, axis=1), compression='gzip')


def test_model_statistics(config, data_loaders, model):
    def get_path_detail():
        return os.path.join(config['folder_out'], config['stage'], 'predict',
                            '{}_m{}_statistics_o{}.h5'.format(phase, mode, observed_views))

    if not os.path.exists(os.path.join(config['folder_out'], config['stage'], 'predict')):
        os.makedirs(os.path.join(config['folder_out'], config['stage'], 'predict'))
    phase_list = [n for n in config['phase_param'] if n not in ['train', 'valid']]
    num_views = config['test_views']
    observe_views_list = [*range(2, 10)]
    for phase in phase_list:
        for mode in [1, 2]:
            for observed_views in observe_views_list:
                path_detail = get_path_detail()
                if os.path.exists(path_detail):
                    os.remove(path_detail)
    path_model = os.path.join(config['folder_out'], config['stage'], 'view_10', config['file_model'])
    checkpoint = torch.load(path_model)
    model.load_state_dict(checkpoint, strict=False)
    model.train(False)
    loss_coef = compute_loss_coef(config)
    for phase in phase_list:
        for mode in [1, 2]:
            for observed_views in observe_views_list:
                query_views = 1
                if mode == 1:
                    K = query_views + 1
                    query_views_index = [math.floor(i / K * (observed_views + query_views) - 1) for i in range(1, K)]
                    observe_views_index = [i for i in range(observed_views + query_views) if i not in query_views_index]
                else:
                    observe_views_index = [*range(observed_views + query_views)][:observed_views]
                    query_views_index = [*range(observed_views + query_views)][observed_views:]
                if mode in [1, 2]:
                    kwargs = {'t_index': observe_views_index, 'q_index': query_views_index}
                else:
                    kwargs = {}
                phase_param = config['phase_param'][phase]
                print('processing data: {}, mode: {}, observe: {}, query: {}'.format(phase, mode, observed_views,
                                                                                     query_views))
                phase_param['num_views'] = num_views
                phase_param['observed_views'] = observed_views
                path_detail = get_path_detail()
                results_all = {'T': {}, 'Q': {}}
                for data in data_loaders[phase]:
                    results = {'T': {}, 'Q': {}}
                    for idx_run in range(config['num_tests']):
                        outputs = model(data, phase_param, loss_coef, single_view=False, test_data=False,
                                        require_results=True, stage=config['stage'],
                                        determine_data=True, **kwargs)
                        for choose in ['T', 'Q']:
                            sub_results = outputs['result'][choose]
                            for key, val in sub_results.items():
                                if key not in ['image', 'recon', 'mask', 'apc', 'shp', 'pres', 'log_ord', 'index']:
                                    continue
                                val = val.data.cpu().numpy()
                                if key in ['image', 'recon', 'mask', 'apc', 'shp']:
                                    val = np.moveaxis(val, -3, -1)
                                if key in results[choose]:
                                    results[choose][key].append(val)
                                else:
                                    results[choose][key] = [val]
                    for choose in ['T', 'Q']:
                        for key, val in results[choose].items():
                            val = np.stack(val)
                            if key in results_all[choose]:
                                results_all[choose][key].append(val)
                            else:
                                results_all[choose][key] = [val]
                with h5py.File(path_detail, 'w') as f:
                    for choose in results_all.keys():
                        f.create_group(choose)
                        for key, val in results_all[choose].items():
                            f[choose].create_dataset(key, data=np.concatenate(val, axis=1), compression='gzip')


def test_model_latent(config, data_loaders, model):
    def get_path_detail():
        return os.path.join(config['folder_out'], config['stage'], 'latent_{}.h5'.format(phase))

    phase_list = [n for n in config['phase_param'] if n not in ['train', 'valid']]
    for phase in phase_list:
        path_detail = get_path_detail()
        if os.path.exists(path_detail):
            os.remove(path_detail)
    path_model = os.path.join(config['folder_out'], config['stage'], 'view_6', config['file_model'])
    checkpoint = torch.load(path_model)
    model.load_state_dict(checkpoint, strict=False)
    model.train(False)
    loss_coef = compute_loss_coef(config)
    for phase in phase_list:
        print('processing {}'.format(phase))
        phase_param = config['phase_param'][phase]
        phase_param['num_views'] = 6
        phase_param['observed_views'] = 6
        path_detail = get_path_detail()
        results_all = {}
        for data in data_loaders[phase]:
            results = {}
            for idx_run in range(config['num_tests']):
                with torch.set_grad_enabled(False):
                    outputs = model(data, phase_param, loss_coef, single_view=False, test_data=True, test_latent=True,
                                    require_results=True, stage='one', determine_data=True)
                    sub_results = outputs['result']['T']
                for key, val in sub_results.items():
                    if key not in ['view_latent', 'lambda', 'attr_obj_latent', 'attr_bck_latent', 'logits_pres',
                                   'matrix', 'index']:
                        continue
                    val = val.data.cpu().numpy()
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
