import argparse

import h5py
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from matplotlib.colors import hsv_to_rgb


# plt.rc('font', size=18, family='serif')


def compute_segre_combine(segre):
    num_colors = segre.shape[0]
    hsv_colors = np.ones((num_colors, 3))
    hsv_colors[:, 0] = (np.linspace(0, 1, num_colors, endpoint=False) + 2 / 3) % 1.0
    segre_colors = hsv_to_rgb(hsv_colors)
    segre_combine = np.clip((segre * segre_colors[:, None, None]).sum(0), 0, 1)
    return segre_combine, segre_colors


def color_spines(ax, color, lw=3):
    for loc in ['top', 'bottom', 'left', 'right']:
        ax.spines[loc].set_linewidth(lw)
        ax.spines[loc].set_color(color)
        ax.spines[loc].set_visible(True)
    return


def plot_image(ax, image, xlabel=None, ylabel=None, border_color=None):
    ax.imshow(image, interpolation='bilinear')
    ax.set_xticks([])
    ax.set_yticks([])
    font = {'family': 'DejaVu Sans',
            'style': 'normal',
            'weight': 'light',
            'size': 18
            }
    ax.set_xlabel(xlabel, color='k', fontdict=font) if xlabel else None
    ax.set_ylabel(ylabel, color='k', fontdict=font) if ylabel else None
    ax.xaxis.set_label_position('top')
    if border_color:
        color_spines(ax, color=border_color)
    return


def plot_null(ax, xlabel=None, ylabel=None):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(xlabel, color='k') if xlabel else None
    ax.set_ylabel(ylabel, color='k') if ylabel else None
    ax.xaxis.set_label_position('top')
    color_spines(ax, color=None)
    return


def select_by_index(x, index_raw):
    x = torch.from_numpy(x)
    index = torch.from_numpy(index_raw)
    x_ndim = x.ndim
    index_ndim = index.ndim
    index = index.reshape(list(index.shape) + [1] * (x_ndim - index_ndim))
    index = index.expand([-1] * index_ndim + list(x.shape[index_ndim:]))
    if index_raw.ndim == 2:
        x_obj = torch.gather(x[:, :-1], index_ndim - 1, index)
        x = torch.cat([x_obj, x[:, -1:]], dim=1)
    elif index_raw.ndim == 3:
        x_obj = torch.gather(x[:, :, :-1], index_ndim - 1, index)
        x = torch.cat([x_obj, x[:, :, -1:]], dim=2)
    else:
        raise AssertionError
    return x.numpy()


def plot_decompose(name, num_views, index, phase='test', scale=1.5):
    folder_data = '../data'
    folder_out = 'metrics'
    name = '{}_v{}'.format(name, num_views)
    path_out = os.path.join(folder_out, name, '{}_{}.h5'.format(phase, num_views))
    with h5py.File(path_out, 'r') as f:
        images = f['image'][0, index:index + 1] / 255
        recon = f['recon'][0, index:index + 1] / 255
        mask = f['mask'][0, index:index + 1] / 255
        shp = f['shp'][0, index:index + 1] / 255
        apc = f['apc'][0, index:index + 1] / 255
        pres = f['pres'][0, index:index + 1] / 255
        order = f['log_ord'][0, index:index + 1]
    score = pres[..., :-1]
    order = np.argsort(-score, axis=-1)
    order_expand = order[:, None].repeat(mask.shape[1], axis=1)
    mask = select_by_index(mask, order_expand)
    shp = select_by_index(shp, order_expand)
    apc = select_by_index(apc, order_expand)
    pres = select_by_index(pres, order)
    obj_slots = apc.shape[2] - 1
    num_cols = num_views
    num_rows = obj_slots + 4
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * scale, num_rows * scale))
    idx_data = 0
    bottom_color = np.ones_like(apc[0, 0, 0])
    bottom_color[:, :, 0] = 0.82745
    bottom_color[:, :, 1] = 0.82745
    bottom_color[:, :, 2] = 0.82745
    for idx_view in range(num_views):
        col = idx_view
        sum_mask = mask.sum(2)
        segre_combine, segre_colors = compute_segre_combine(mask[idx_data, idx_view])
        plot_image(axes[0, col], images[idx_data, idx_view], ylabel='image' if col == 0 else None)
        plot_image(axes[1, col], recon[idx_data, idx_view], ylabel='recon' if col == 0 else None)
        plot_image(axes[2, col], segre_combine, ylabel='segment' if col == 0 else None)
        row = 3
        for idx_obj in range(apc.shape[2]):
            if obj_slots <= idx_obj < apc.shape[2] - 1:
                continue
            vis_shp = shp[idx_data, idx_view, idx_obj] * pres[idx_data, idx_obj, None, None, None]
            vis_shp = (vis_shp > 0.5).astype(np.float32)
            img_obj = vis_shp * apc[idx_data, idx_view, idx_obj] + (1 - vis_shp) * bottom_color
            border_color = tuple(segre_colors[idx_obj])
            if pres[idx_data, idx_obj] < 0.5:
                border_color = None
            ylabel = 'obj {}'.format(idx_obj + 1) if idx_obj < apc.shape[2] - 1 else 'bck'
            plot_image(axes[row, col], img_obj, border_color=border_color, ylabel=ylabel if col == 0 else None)
            row += 1
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='shop_multi_complex_10')
    parser.add_argument('--num_views', type=int, default=4)
    parser.add_argument('--phase', type=str, default='general')
    args = parser.parse_args()
    folder = os.path.join('visualization', 'decompose', '{}_v{}'.format(args.name, args.num_views), args.phase)
    if not os.path.exists(folder):
        os.makedirs(folder)
    for index in range(100):
        fig = plot_decompose(args.name, args.num_views, index, phase=args.phase)
        fig.savefig(os.path.join(folder, '{}.jpg'.format(index)), bbox_inches='tight', pad_inches=0)
        plt.close()


if __name__ == '__main__':
    main()
