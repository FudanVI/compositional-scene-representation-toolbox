import math
import os
import time
from typing import Dict, List, Tuple, Union

import h5py
import hydra
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import DataPrefetcher
from imageio import imwrite
from omegaconf import DictConfig
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter


def get_loss_coef(
    cfg: DictConfig,
    step: Union[int, None] = None,
) -> Dict[str, torch.Tensor]:
    if step is None:
        step = cfg.run_training.max_steps
    loss_coef = {
        'nll': 1.0,
        'kld_view': 1.0,
        'kld_bck': 1.0,
        'kld_obj': 1.0,
        'kld_pres': 1.0,
    }
    for key, val in cfg.run_training.loss.coef.items():
        step_list = [1] + val['step'] + [cfg.run_training.max_steps]
        assert len(step_list) == len(val['value'])
        assert len(step_list) == len(val['linear']) + 1
        assert step_list == sorted(step_list)
        for idx in range(len(step_list) - 1):
            if step <= step_list[idx + 1]:
                diff_y = step - step_list[idx]
                diff_x = step_list[idx + 1] - step_list[idx]
                ratio = diff_y / diff_x
                val_1 = val['value'][idx]
                val_2 = val['value'][idx + 1]
                if val['linear'][idx]:
                    loss_coef[key] = (1 - ratio) * val_1 + ratio * val_2
                else:
                    loss_coef[key] = math.exp((1 - ratio) * math.log(val_1) +
                                              ratio * math.log(val_2))
                assert math.isfinite(loss_coef[key])
                break
        else:
            raise ValueError
    for name in ['kld', 'reg']:
        if f'sched_{name}' in loss_coef:
            coef = loss_coef[f'sched_{name}']
            for key in loss_coef:
                if key.split('_')[0] == name:
                    loss_coef[key] *= coef
    loss_coef = {
        key: torch.tensor(val, pin_memory=True).cuda(non_blocking=True)
        for key, val in loss_coef.items()
    }
    return loss_coef


def get_path_ckpt(
    cfg: DictConfig,
    rank: int,
    create_folder: bool = False,
) -> Dict[str, str]:
    if rank == 0 and create_folder and not os.path.exists(cfg.folder_ckpt):
        os.mkdir(cfg.folder_ckpt)
    torch.distributed.barrier()
    path_ckpt = {
        key: os.path.join(cfg.folder_ckpt, f'{key}.pth')
        for key in ['train_prev', 'train_curr', 'best_prev', 'best_curr']
    }
    return path_ckpt


def get_optimizer_scheduler(
    cfg: DictConfig,
    model: DistributedDataParallel,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    def lr_lambda(x):
        decay_rate = cfg.run_training.scheduler.lr_decay
        decay_steps = cfg.run_training.scheduler.decay_steps
        warmup_steps = cfg.run_training.scheduler.warmup_steps
        decay_ratio = 0 if decay_steps == 0 else x / decay_steps
        decay_coef = pow(decay_rate, decay_ratio)
        warmup_ratio = 1 if warmup_steps == 0 else x / warmup_steps
        warmup_coef = min(warmup_ratio, 1)
        coef = decay_coef * warmup_coef
        return coef

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=cfg.run_training.optimizer.lr,
        fused=True,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lr_lambda,
    )
    return optimizer, scheduler


def run_training(
    cfg: DictConfig,
    model: DistributedDataParallel,
    data_prefetchers: Dict[str, DataPrefetcher],
) -> None:
    @torch.no_grad()
    def log_scalar(
        losses: Dict[str, torch.Tensor],
        metrics: Dict[str, torch.Tensor],
        phase: str,
    ) -> None:
        world_size = torch.distributed.get_world_size()
        for scalars in [losses, metrics]:
            for key, val in scalars.items():
                torch.distributed.reduce(
                    val,
                    dst=0,
                )
                scalars[key] /= world_size
        scalars = {
            **{f'loss_{key}': val
               for key, val in losses.items()},
            **{f'metric_{key}': val
               for key, val in metrics.items()},
        }
        if rank == 0:
            for key, val in scalars.items():
                writer.add_scalar(
                    f'{phase}/{key}',
                    scalar_value=val,
                    global_step=idx_step,
                )
        return

    @torch.no_grad()
    def log_overview(
        outputs: Dict[str, torch.Tensor],
        phase: str,
        dpi: int = 100,
    ) -> None:
        def plot_image(ax, image, xlabel=None, ylabel=None, color=None):
            plot = ax.imshow(image, interpolation='bilinear')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(xlabel,
                          color='k' if color is None else color,
                          fontfamily='monospace') if xlabel else None
            ax.set_ylabel(ylabel,
                          color='k' if color is None else color,
                          fontfamily='monospace') if ylabel else None
            ax.xaxis.set_label_position('top')
            return plot

        def cvt_image(image):
            if image.shape[-1] == 1:
                image = np.repeat(image, 3, axis=-1)
            return image

        def get_overview(x_all, idx_fig):
            sub_x_all = {key: val[idx_fig] for key, val in x_all.items()}
            rows, cols = 4 * num_views, num_slots + 2
            figsize = (cols, (rows + 0.5 / num_views) * image_ht / image_wd)
            fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)
            for idx_view in range(num_views):
                xlabel = 'scene' if idx_view == 0 else None
                plot_image(
                    axes[idx_view * 4, 0],
                    cvt_image(sub_x_all['image'][idx_view]),
                    xlabel=xlabel,
                )
                plot_image(
                    axes[idx_view * 4 + 1, 0],
                    cvt_image(sub_x_all['recon'][idx_view]),
                )
                plot_image(
                    axes[idx_view * 4 + 2, 0],
                    cvt_image(sub_x_all['recon_aux'][idx_view]),
                )
                plot_image(
                    axes[idx_view * 4 + 3, 0],
                    cvt_image(sub_x_all['recon_aux_imp'][idx_view]),
                )
                for idx_slot in range(num_slots):
                    if idx_view == 0:
                        xlabel = 'obj_{}'.format(idx_slot)
                    else:
                        xlabel = None
                    if sub_x_all['pres'][idx_slot] >= 0.5:
                        color = [1.0, 0.5, 0.0]
                    else:
                        color = [0.0, 0.5, 1.0]
                    plot_image(
                        axes[idx_view * 4, idx_slot + 1],
                        cvt_image(sub_x_all['apc'][idx_view, idx_slot]),
                        xlabel=xlabel,
                        color=color,
                    )
                    plot_image(
                        axes[idx_view * 4 + 1, idx_slot + 1],
                        cvt_image(sub_x_all['apc_aux'][idx_view, idx_slot]),
                    )
                    plot_image(
                        axes[idx_view * 4 + 2, idx_slot + 1],
                        cvt_image(sub_x_all['shp'][idx_view, idx_slot]),
                    )
                    plot_image(
                        axes[idx_view * 4 + 3, idx_slot + 1],
                        cvt_image(sub_x_all['sdw_apc'][idx_view, idx_slot]),
                    )
                    for offset in [2]:
                        axes[idx_view * 4 + offset, idx_slot + 1].add_patch(
                            plt.Circle((
                                sub_x_all['trs'][idx_view, idx_slot][1],
                                sub_x_all['trs'][idx_view, idx_slot][0],
                            ),
                                       image_wd / 64,
                                       color='r'), )
                xlabel = 'bck' if idx_view == 0 else None
                plot_image(
                    axes[idx_view * 4, -1],
                    cvt_image(sub_x_all['bck_imp'][idx_view]),
                    xlabel=xlabel,
                )
                plot_image(
                    axes[idx_view * 4 + 1, -1],
                    cvt_image(sub_x_all['bck'][idx_view]),
                )
                plot_image(
                    axes[idx_view * 4 + 2, -1],
                    cvt_image(sub_x_all['mask_bck'][idx_view]),
                )
                for offset in range(3, 4):
                    axes[idx_view * 4 + offset, -1].set_visible(False)
            fig.tight_layout(pad=0)
            fig.canvas.draw()
            out_wd, out_ht = fig.canvas.get_width_height()
            out = np.frombuffer(fig.canvas.tostring_rgb(),
                                dtype='uint8').reshape(out_ht, out_wd, -1)
            plt.close(fig)
            return out

        if outputs['image'].shape[0] < cfg.run_training.log_image_count:
            if rank == 0:
                x_all = {
                    key: [
                        torch.empty(
                            val.shape,
                            dtype=val.dtype,
                            device=val.device,
                        ) for _ in range(world_size)
                    ]
                    for key, val in outputs.items()
                }
            else:
                x_all = {key: None for key in outputs}
            for key, val in outputs.items():
                torch.distributed.gather(val, x_all[key], dst=0)
            if rank == 0:
                x_all = {
                    key: torch.cat(val, dim=0)
                    for key, val in x_all.items()
                }
        else:
            if rank == 0:
                x_all = outputs
        if rank == 0:
            x_all = {
                key: val[:cfg.run_training.log_image_count]
                for key, val in x_all.items()
            }
            batch_size, num_views, num_slots, image_ht, image_wd = \
                x_all['apc'].shape[:5]
            key_list = [
                'image', 'recon', 'recon_aux', 'recon_aux_imp', 'bck',
                'bck_imp', 'apc', 'apc_aux', 'sdw_apc'
            ]
            for key in key_list:
                x_all[key] = torch.clamp((x_all[key] + 1) * 0.5, 0, 1)
            trs = x_all['trs']
            coef_trs = torch.tensor(
                [image_ht, image_wd],
                dtype=trs.dtype,
                device=trs.device,
            )[None, None, None]
            x_all['trs'] = (trs + 1) * 0.5 * coef_trs
            white = torch.ones(
                [1, 1, 1, 1, 1, 3],
                dtype=trs.dtype,
                device=trs.device,
            )
            green = torch.tensor(
                [0, 1, 0],
                dtype=trs.dtype,
                device=trs.device,
            )[None, None, None, None, None]
            x_all['shp'] = x_all['shp'] * white + \
                (1 - x_all['shp']) * x_all['shp_imp'] * green
            x_all['sdw_apc'] = x_all['sdw_shp'] * x_all['sdw_apc'] + \
                (1 - x_all['sdw_shp']) * x_all['bck_imp'][:, :, None]
            x_all = {
                key: val.to(torch.float32).cpu().numpy()
                for key, val in x_all.items()
            }
            overview_list = [
                get_overview(x_all, idx) for idx in range(batch_size)
            ]
            overview = np.concatenate(overview_list, axis=0)
            folder_out = f'overview_{phase}'
            if not os.path.exists(folder_out):
                os.mkdir(folder_out)
            imwrite(os.path.join(folder_out, f'step_{idx_step}.png'), overview)
        return

    @torch.no_grad()
    def save_log_train(
        data: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
        losses: Dict[str, torch.Tensor],
    ) -> None:
        metrics = model.module.compute_metrics(data, outputs)
        log_scalar(losses, metrics, phase='train')
        return

    @torch.no_grad()
    def save_log_val(
        outputs_train: Dict[str, torch.Tensor],
        best_step: int,
        best_loss: float,
        overview_key_list: List[str] = [
            'image', 'recon', 'recon_aux', 'recon_aux_imp', 'mask_bck', 'bck',
            'bck_imp', 'trs', 'pres', 'apc', 'apc_aux', 'shp', 'shp_imp',
            'sdw_apc', 'sdw_shp'
        ],
    ) -> Tuple[int, float]:
        # Log training overview
        outputs_train = {
            key: val
            for key, val in outputs_train.items() if key in overview_key_list
        }
        log_overview(outputs_train, phase='train')

        # Log validation
        data_prefetchers['val'].setup()
        loss_coef = get_loss_coef(cfg)
        num_data = 0
        while True:
            data = data_prefetchers['val'].next()
            if data is None:
                break
            with torch.cuda.amp.autocast(
                    dtype=torch.bfloat16,
                    enabled=(cfg.precision == 'bf16'),
            ):
                outputs = model(
                    data['image'],
                    loss_coef['temp_pres'],
                    loss_coef['temp_shp'],
                    loss_coef['noise_scale'],
                    loss_coef['noise_min'],
                    loss_coef['noise_max'],
                    loss_coef['ratio_stick_breaking'],
                    cfg.run_training.num_slots['val'],
                )
                losses = model.module.compute_losses(outputs, loss_coef)
                metrics = model.module.compute_metrics(data, outputs)
            batch_size = data['image'].shape[0]
            if num_data == 0:
                outputs_all = {
                    key: val
                    for key, val in outputs.items() if key in overview_key_list
                }
                losses_all = {
                    key: val * batch_size
                    for key, val in losses.items()
                }
                metrics_all = {
                    key: val * batch_size
                    for key, val in metrics.items()
                }
            else:
                if num_data < cfg.run_training.log_image_count:
                    for key, val in outputs_all.items():
                        outputs_all[key] = torch.cat(
                            [val, outputs[key]],
                            dim=0,
                        )
                for key, val in losses.items():
                    losses_all[key] += val * batch_size
                for key, val in metrics.items():
                    metrics_all[key] += val * batch_size
            num_data += batch_size
        losses_all = {key: val / num_data for key, val in losses_all.items()}
        metrics_all = {key: val / num_data for key, val in metrics_all.items()}
        log_scalar(losses_all, metrics_all, phase='val')
        log_overview(outputs_all, phase='val')

        # Save checkpoints and print metrics
        if rank == 0:
            curr_loss = losses_all['opt'].item()
            if curr_loss < best_loss:
                best_step = idx_step
                best_loss = curr_loss
                is_best = True
            else:
                is_best = False
            save_dict = {
                'idx_epoch': idx_epoch,
                'idx_batch': idx_batch,
                'idx_step': idx_step,
                'best_step': best_step,
                'best_loss': best_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if is_best:
                if os.path.exists(path_ckpt['best_curr']):
                    os.rename(path_ckpt['best_curr'], path_ckpt['best_prev'])
                torch.save(save_dict, path_ckpt['best_curr'])
            if os.path.exists(path_ckpt['train_curr']):
                os.rename(path_ckpt['train_curr'], path_ckpt['train_prev'])
            torch.save(save_dict, path_ckpt['train_curr'])
            metric_text_list = [
                'ARI_ALL_S {:.3f}'.format(metrics_all['ari_all_s']),
                'ARI_ALL_M {:.3f}'.format(metrics_all['ari_all_m']),
                'ARI_OBJ_S {:.3f}'.format(metrics_all['ari_obj_s']),
                'ARI_OBJ_M {:.3f}'.format(metrics_all['ari_obj_m']),
                'MSE: {:.2e}'.format(metrics_all['mse']),
                'Count {:.3f}'.format(metrics_all['count']),
            ]
            print(f'Step: {idx_step}/{cfg.run_training.max_steps}')
            print((' ' * 4).join(metric_text_list))
            print('Steps per sec: {:.2f}'.format(
                cfg.run_training.log_val_interval / duration))
            print(f'Best Step: {best_step}')
            print()
        torch.distributed.barrier()
        return best_step, best_loss

    def initialize_variables() -> Tuple[int, int, int, int, float]:
        idx_epoch = 0
        idx_batch = 0
        idx_step = 0
        best_step = 0
        best_loss = float('inf')
        if rank == 0:
            print('Start training')
        return idx_epoch, idx_batch, idx_step, best_step, best_loss

    def load_variables(key: str) -> Tuple[int, int, int, int, float]:
        load_dict = torch.load(path_ckpt[key])
        idx_epoch = load_dict['idx_epoch']
        idx_batch = load_dict['idx_batch'] + 1
        idx_step = load_dict['idx_step']
        best_step = load_dict['best_step']
        best_loss = load_dict['best_loss']
        model.load_state_dict(load_dict['model_state_dict'])
        optimizer.load_state_dict(load_dict['optimizer_state_dict'])
        return idx_epoch, idx_batch, idx_step, best_step, best_loss

    def load_from_checkpoint() -> Tuple[int, int, int, int, float]:
        try:
            variables_best = load_variables('best_curr')
        except:
            try:
                idx_epoch, idx_batch, idx_step, best_step, best_loss = \
                    load_variables('best_prev')
            except:
                if rank == 0:
                    for key in path_ckpt:
                        if os.path.exists(path_ckpt[key]):
                            os.remove(path_ckpt[key])
                torch.distributed.barrier()
                idx_epoch, idx_batch, idx_step, best_step, best_loss = \
                    initialize_variables()
            else:
                if rank == 0:
                    os.rename(path_ckpt['best_prev'], path_ckpt['best_curr'])
                    for key in ['train_prev', 'train_curr']:
                        if os.path.exists(path_ckpt[key]):
                            os.remove(path_ckpt[key])
                    print(f'Resume from previous best: {idx_step}')
                torch.distributed.barrier()
        else:
            load_from_best = False
            try:
                idx_epoch, idx_batch, idx_step, best_step, best_loss = \
                    load_variables('train_curr')
            except:
                try:
                    idx_epoch, idx_batch, idx_step, best_step, best_loss = \
                        load_variables('train_prev')
                except:
                    load_from_best = True
                else:
                    if rank == 0:
                        os.rename(path_ckpt['train_prev'],
                                  path_ckpt['train_curr'])
                    torch.distributed.barrier()
                    if idx_step >= variables_best[1]:
                        if rank == 0:
                            print(f'Resume from previous latest: {idx_step}')
                    else:
                        load_from_best = True
            else:
                if idx_step >= variables_best[1]:
                    if rank == 0:
                        print(f'Resume from current latest: {idx_step}')
                else:
                    load_from_best = True
            if load_from_best:
                if rank == 0:
                    for key in ['train_prev', 'train_curr']:
                        if os.path.exists(path_ckpt[key]):
                            os.remove(path_ckpt[key])
                torch.distributed.barrier()
                idx_epoch, idx_batch, idx_step, best_step, best_loss = \
                    variables_best
                if rank == 0:
                    print(f'Resume from current best: {idx_step}')
        return idx_epoch, idx_batch, idx_step, best_step, best_loss

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    optimizer, scheduler = get_optimizer_scheduler(cfg, model)
    if cfg.profiling:
        if rank == 0:
            print('Start profiling')
        data_prefetchers['val'].setup()
        for idx_batch in range(10):
            data = data_prefetchers['val'].next()
            if data is None:
                break
            loss_coef = get_loss_coef(cfg, idx_batch)
            torch.cuda.nvtx.range_push(f'batch_{idx_batch}')
            with torch.cuda.amp.autocast(
                    dtype=torch.bfloat16,
                    enabled=(cfg.precision == 'bf16'),
            ):
                torch.cuda.nvtx.range_push('forward')
                outputs = model(
                    data['image'],
                    loss_coef['temp_pres'],
                    loss_coef['temp_shp'],
                    loss_coef['noise_scale'],
                    loss_coef['noise_min'],
                    loss_coef['noise_max'],
                    loss_coef['ratio_stick_breaking'],
                    cfg.run_training.num_slots['train'],
                )
                torch.cuda.nvtx.range_pop()
                torch.cuda.nvtx.range_push('loss')
                losses = model.module.compute_losses(outputs, loss_coef)
                torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push('backward')
            optimizer.zero_grad()
            losses['opt'].backward()
            optimizer.step()
            scheduler.step()
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_pop()
        exit()
    if 'folder_pretrain' in cfg:
        folder_pretrain = hydra.utils.to_absolute_path(cfg.folder_pretrain)
        load_dict = torch.load(
            os.path.join(folder_pretrain, cfg.folder_ckpt, 'best_curr.pth'))
        model.load_state_dict(load_dict['model_state_dict'])
    path_ckpt = get_path_ckpt(cfg, rank, create_folder=True)
    if not cfg.resume:
        idx_epoch, idx_batch, idx_step, best_step, best_loss = \
            initialize_variables()
    else:
        idx_epoch, idx_batch, idx_step, best_step, best_loss = \
            load_from_checkpoint()
    if rank == 0:
        print()
    with SummaryWriter('tensorboard', purge_step=idx_step + 1) as writer:
        data_prefetchers['train'].setup(idx_epoch, idx_batch)
        time_s = time.time()
        stop_training = idx_step >= cfg.run_training.max_steps
        while not stop_training:
            while True:
                data = data_prefetchers['train'].next()
                if data is None:
                    break
                idx_step += 1
                if idx_step > cfg.run_training.max_steps:
                    stop_training = True
                    break
                loss_coef = get_loss_coef(cfg, idx_step)
                with torch.cuda.amp.autocast(
                        dtype=torch.bfloat16,
                        enabled=(cfg.precision == 'bf16'),
                ):
                    outputs = model(
                        data['image'],
                        loss_coef['temp_pres'],
                        loss_coef['temp_shp'],
                        loss_coef['noise_scale'],
                        loss_coef['noise_min'],
                        loss_coef['noise_max'],
                        loss_coef['ratio_stick_breaking'],
                        cfg.run_training.num_slots['train'],
                    )
                    losses = model.module.compute_losses(outputs, loss_coef)
                optimizer.zero_grad()
                losses['opt'].backward()
                if 'ratio_dec' in loss_coef:
                    with torch.no_grad():
                        for param in model.module.enc.net_feat.parameters():
                            param.grad *= loss_coef['ratio_dec']
                        for param in model.module.dec.parameters():
                            param.grad *= loss_coef['ratio_dec']
                grad_clip_norm = cfg.run_training.optimizer.gradient_clip_norm
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   grad_clip_norm)
                optimizer.step()
                scheduler.step()
                if idx_step % cfg.run_training.log_train_interval == 0:
                    save_log_train(data, outputs, losses)
                if idx_step % cfg.run_training.log_val_interval == 0:
                    time_e = time.time()
                    duration = time_e - time_s
                    best_step, best_loss = \
                        save_log_val(outputs, best_step, best_loss)
                    time_s = time.time()
                idx_batch += 1
            idx_epoch += 1
            idx_batch = 0
            data_prefetchers['train'].setup(idx_epoch, idx_batch)
    return


@torch.no_grad()
def run_testing(
    cfg: DictConfig,
    model: DistributedDataParallel,
    data_prefetchers: Dict[str, DataPrefetcher],
) -> None:
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    path_ckpt = get_path_ckpt(cfg, rank, create_folder=False)
    load_dict = torch.load(path_ckpt['best_curr'])
    model.load_state_dict(load_dict['model_state_dict'])
    loss_coef = get_loss_coef(cfg)
    for phase in data_prefetchers:
        data_prefetchers[phase].setup()
        if rank == 0:
            results_all = None
        while True:
            data = data_prefetchers[phase].next()
            if data is None:
                break
            sub_results = None
            for _ in range(cfg.run_testing.num_tests):
                with torch.cuda.amp.autocast(
                        dtype=torch.bfloat16,
                        enabled=(cfg.precision == 'bf16'),
                ):
                    outputs = model(
                        data['image'],
                        loss_coef['temp_pres'],
                        loss_coef['temp_shp'],
                        loss_coef['noise_scale'],
                        loss_coef['noise_min'],
                        loss_coef['noise_max'],
                        loss_coef['ratio_stick_breaking'],
                        cfg.run_testing.num_slots[phase],
                    )
                    outputs = model.module.convert_outputs(outputs)
                if sub_results is None:
                    sub_results = {key: [val] for key, val in outputs.items()}
                else:
                    for key, val in outputs.items():
                        sub_results[key].append(val)
            sub_results = {
                key: torch.stack(val, dim=0)
                for key, val in sub_results.items()
            }
            for key, val in sub_results.items():
                if val.dtype in [torch.float16, torch.bfloat16]:
                    sub_results[key] = val.to(dtype=torch.float32)
            if rank == 0:
                results = {
                    key: [
                        torch.empty(
                            val.shape,
                            dtype=val.dtype,
                            device=val.device,
                        ) for _ in range(world_size)
                    ]
                    for key, val in sub_results.items()
                }
            else:
                results = {key: None for key in sub_results}
            for key, val in sub_results.items():
                torch.distributed.gather(val, results[key], dst=0)
            if rank == 0:
                results = {
                    key: torch.cat(val, dim=1).cpu().numpy()
                    for key, val in results.items()
                }
                if results_all is None:
                    results_all = {key: [val] for key, val in results.items()}
                else:
                    for key, val in results.items():
                        results_all[key].append(val)
        if rank == 0:
            num_data = len(data_prefetchers[phase].dataloader.dataset)
            results_all = {
                key: np.concatenate(val, axis=1)[:, :num_data]
                for key, val in results_all.items()
            }
            with h5py.File(f'{phase}.h5', 'w') as f:
                for key, val in results_all.items():
                    f.create_dataset(key, data=val, compression='gzip')
        torch.distributed.barrier()
    return
