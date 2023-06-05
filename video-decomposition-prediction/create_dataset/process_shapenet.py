import os

import matplotlib.pyplot as plt
import sunds
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"


def color_spines(ax, color, lw=3):
    for loc in ['top', 'bottom', 'left', 'right']:
        ax.spines[loc].set_linewidth(lw)
        ax.spines[loc].set_color(color)
        ax.spines[loc].set_visible(True)


def plot_image(ax, image, xlabel=None, ylabel=None, border_color=None, gray=False):
    if gray:
        ax.imshow(image, interpolation='bilinear', cmap='gray')
    else:
        ax.imshow(image, interpolation='bilinear')
    ax.set_xticks([])
    ax.set_yticks([])
    font = {'family': 'DejaVu Sans',
            'style': 'normal',
            'weight': 'light',
            'size': 14
            }
    ax.set_xlabel(xlabel, color='k', fontdict=font) if xlabel else None
    ax.set_ylabel(ylabel, color='k', fontdict=font) if ylabel else None
    ax.xaxis.set_label_position('top')
    if border_color:
        color_spines(ax, color=border_color)
    return


def main():
    builder = sunds.builder('multi_shapenet_conditional', data_dir='/home/gcm/tensorflow_datasets')
    ds = builder.as_dataset(
        split='test',
        # Stack all camera of a scene together
        task=sunds.tasks.Scenes(),
    )
    print(ds.element_spec)
    idx = 0
    for ex in ds:
        print("processing {}".format(idx))
        num_rows = 1
        num_cols = 10
        scale = 2
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * scale, num_rows * scale))
        for i in range(10):
            segment = ex['cameras']['camera_{}'.format(i)]['instance_image'].numpy()
            # total_segment = segment.reshape(-1)
            # color_num = []
            # for pixel in total_segment:
            #     if str(pixel) not in color_num:
            #         color_num.append(str(pixel))
            # print(color_num)
            plot_image(axes[i], segment, gray=True)
        fig.savefig('/home/gcm/save/shapenet_conditional/segment_{}.jpg'.format(idx), bbox_inches='tight', pad_inches=0)
        plt.close()
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * scale, num_rows * scale))
        for i in range(10):
            image = ex['cameras']['camera_{}'.format(i)]['color_image'].numpy()
            plot_image(axes[i], image)
        fig.savefig('/home/gcm/save/shapenet_conditional/image_{}.jpg'.format(idx), bbox_inches='tight', pad_inches=0)
        plt.close()
        idx += 1


if __name__ == '__main__':
    main()
