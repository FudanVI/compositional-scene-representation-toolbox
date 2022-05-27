import argparse
import numpy as np
import os
from imageio import imread
from common import load_config, save_dataset


def read_and_crop(config, path):
    image = imread(path)
    image = image[config['crop_up']:config['crop_down'], config['crop_left']:config['crop_right']]
    return image


def compute_values(shp, segment, max_objects, th=0.5):
    scores = shp[..., -1] * np.exp(-np.square(shp[..., :-1] - segment[None, ..., :-1]).sum(-1))
    segment_obj = np.argmax(scores, axis=0)
    mask_bck = segment[..., -1] < th
    segment = (mask_bck * max_objects + (1 - mask_bck) * segment_obj).astype(np.uint8)
    overlap = (shp[..., -1] >= th).sum(0).astype(np.uint8)
    masks = (shp[..., -1:] * 255).astype(np.uint8)
    zeros = np.zeros([max_objects - masks.shape[0], *masks.shape[1:]], dtype=np.uint8)
    ones = np.full([1, *masks.shape[1:]], 255, dtype=np.uint8)
    masks = np.concatenate([masks, zeros, ones])
    return segment, overlap, masks


def generate_data_single(config, folder, max_objects):
    filenames_all = os.listdir(folder)
    filenames_shp = [val for val in filenames_all if val.split('_')[0] == 'mask']
    filenames_shp = sorted(filenames_shp, key=lambda x: int(x.split('.')[0].split('_')[-1]))
    image = read_and_crop(config, os.path.join(folder, 'image.png'))[..., :-1]
    segment = read_and_crop(config, os.path.join(folder, 'segment.png')) / 255
    shp = np.stack([read_and_crop(config, os.path.join(folder, val)) / 255 for val in filenames_shp])
    segment, overlap, masks = compute_values(shp, segment, max_objects)
    data = {'image': image, 'segment': segment, 'overlap': overlap, 'masks': masks}
    return data


def generate_data_multi(config, folder, max_objects):
    num_views = config['num_views']
    filenames_all = os.listdir(folder)
    filenames_shp_list = [[] for _ in range(num_views)]
    for filename in filenames_all:
        filename_split = filename.split('_')
        if filename_split[0] == 'mask':
            filenames_shp_list[int(filename_split[1])].append(filename)
    data_list = {key: [] for key in ['image', 'segment', 'overlap', 'masks']}
    for idx_view in range(num_views):
        image = read_and_crop(config, os.path.join(folder, 'image_{}.png'.format(idx_view)))[..., :-1]
        segment = read_and_crop(config, os.path.join(folder, 'segment_{}.png'.format(idx_view))) / 255
        filenames_shp = sorted(filenames_shp_list[idx_view], key=lambda x: int(x.split('.')[0].split('_')[-1]))
        shp = np.stack([read_and_crop(config, os.path.join(folder, val)) / 255 for val in filenames_shp])
        segment, overlap, masks = compute_values(shp, segment, max_objects)
        data_list['image'].append(image)
        data_list['segment'].append(segment)
        data_list['overlap'].append(overlap)
        data_list['masks'].append(masks)
    data = {key: np.stack(val) for key, val in data_list.items()}
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name')
    parser.add_argument('--path_config')
    parser.add_argument('--folder_train')
    parser.add_argument('--folder_general')
    parser.add_argument('--folder_out')
    parser.add_argument('--multiview', action='store_true')
    config = load_config(parser)
    sub_folders_train = sorted(os.listdir(config['folder_train']))
    sub_folders_general = sorted(os.listdir(config['folder_general']))
    folders_all = {
        'train': (config['folder_train'], sub_folders_train[:config['split_train']]),
        'valid': (config['folder_train'], sub_folders_train[config['split_train']:config['split_valid']]),
        'test': (config['folder_train'], sub_folders_train[config['split_valid']:]),
        'general': (config['folder_general'], sub_folders_general),
    }
    folders_all = {key: [os.path.join(val[0], sub_val) for sub_val in val[1]] for key, val in folders_all.items()}
    max_objects_all = {
        'train': config['max_objects_train'],
        'valid': config['max_objects_train'],
        'test': config['max_objects_train'],
        'general': config['max_objects_general'],
    }
    generate_data = generate_data_multi if config['multiview'] else generate_data_single
    datasets = {}
    for phase, folders in folders_all.items():
        data_list = []
        for folder in folders:
            data = generate_data(config, folder, max_objects_all[phase])
            data_list.append(data)
        datasets[phase] = data_list
    save_dataset(config, datasets)
    return


if __name__ == '__main__':
    main()
