import argparse
import json
import os
import pdb

import numpy as np
from imageio import imread

from common import load_config, save_dataset, resize_image


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


def generate_data_single(config, folder, max_objects, class_map=None):
    filenames_all = os.listdir(folder)
    filenames_shp = [val for val in filenames_all if val.split('_')[0] == 'mask']
    filenames_shp = sorted(filenames_shp, key=lambda x: int(x.split('.')[0].split('_')[-1]))
    image = read_and_crop(config, os.path.join(folder, 'image.png'))[..., :-1]
    segment = read_and_crop(config, os.path.join(folder, 'segment.png'))
    image = resize_image(image, (config['img_size'], config['img_size']))
    segment = resize_image(segment, (config['img_size'], config['img_size']))
    shp = np.stack(
        [resize_image(read_and_crop(config, os.path.join(folder, val)), (config['img_size'], config['img_size'])) for
         val in filenames_shp])
    segment = segment / 255
    shp = shp / 255
    # if image.shape[0] != config['img_size']:
    #     image = torch.FloatTensor(image)[None].permute(0, 3, 1, 2)
    #     segment = torch.FloatTensor(segment)[None].permute(0, 3, 1, 2)
    #     shp = torch.FloatTensor(shp).permute(0, 3, 1, 2)
    #     image = F.interpolate(image, size=config['img_size']).permute(0, 2, 3, 1).squeeze().numpy().astype(np.uint8)
    #     segment = F.interpolate(segment, size=config['img_size']).permute(0, 2, 3, 1).squeeze().numpy()
    #     shp = F.interpolate(shp, size=config['img_size']).permute(0, 2, 3, 1).squeeze().numpy()
    segment, overlap, masks = compute_values(shp, segment, max_objects)
    data = {'image': image, 'segment': segment, 'overlap': overlap, 'masks': masks}
    if class_map is not None:
        with open(os.path.join(folder, 'config.json'), 'r') as f:
            classes = json.load(f)
        classes = [class_map[cls] for cls in classes['class']]
        obj_present = len(classes)
        classes += (max_objects - obj_present) * [-1]
        data.update({'cls': np.array(classes, dtype=np.int64)})
    return data


def generate_data_multi(config, folder, max_objects, class_map=None):
    num_views = config['num_views']
    ratio = config['current_num_views'] // num_views
    view_index = [ratio * i for i in range(num_views)]
    filenames_all = os.listdir(folder)
    filenames_shp_list = [[] for _ in range(num_views)]
    for filename in filenames_all:
        filename_split = filename.split('_')
        if filename_split[0] == 'mask':
            if int(filename_split[1]) % ratio == 0:
                index = int(filename_split[1]) // ratio
                filenames_shp_list[index].append(filename)
    data_list = {key: [] for key in ['image', 'segment', 'overlap', 'masks']}
    label_list = {key: [] for key in ['viewpoint']}
    for idx_view in range(num_views):
        image = read_and_crop(config, os.path.join(folder, 'image_{}.png'.format(view_index[idx_view])))[..., :-1]
        segment = read_and_crop(config, os.path.join(folder, 'segment_{}.png'.format(view_index[idx_view])))
        image = resize_image(image, (config['img_size'], config['img_size']))
        segment = resize_image(segment, (config['img_size'], config['img_size']))
        filenames_shp = sorted(filenames_shp_list[idx_view], key=lambda x: int(x.split('.')[0].split('_')[-1]))
        shp = np.stack(
            [resize_image(read_and_crop(config, os.path.join(folder, val)), (config['img_size'], config['img_size']))
             for val in filenames_shp])
        segment = segment / 255
        shp = shp / 255
        segment, overlap, masks = compute_values(shp, segment, max_objects)
        data_list['image'].append(image)
        data_list['segment'].append(segment)
        data_list['overlap'].append(overlap)
        data_list['masks'].append(masks)
        with open(os.path.join(folder, 'view_{}.json'.format(view_index[idx_view])), 'r') as f:
            view_map = json.load(f)
            theta = view_map['theta']
            phi = view_map['phi']
            rho = view_map['rho']
            label_list['viewpoint'].append(
                np.array([np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi), rho], dtype=np.float32))
    data = {key: np.stack(val) for key, val in data_list.items()}
    label = {key: np.stack(val) for key, val in label_list.items()}
    if class_map is not None:
        with open(os.path.join(folder, 'category.json'), 'r') as f:
            classes = json.load(f)
        classes = [class_map[cls] for cls in classes['category']]
        obj_present = len(classes)
        classes += (max_objects - obj_present) * [-1]
        data.update({'cls': np.array(classes, dtype=np.int64)})
    return data, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name')
    parser.add_argument('--path_config')
    parser.add_argument('--folder_train')
    parser.add_argument('--folder_general')
    parser.add_argument('--folder_out')
    parser.add_argument('--add_class', action='store_true')
    parser.add_argument('--multiview', action='store_true')
    config = load_config(parser)
    label_name = config['name'] + '_' + str(config['num_views']) + '_' + 'viewpoint'
    config['name'] = config['name'] + '_' + str(config['num_views'])

    sub_folders_train = sorted(os.listdir(config['folder_train']))[:config['split_test']]
    sub_folders_general = sorted(os.listdir(config['folder_general']))
    folders_all = {
        'train': (config['folder_train'], sub_folders_train[:config['split_train']]),
        'valid': (config['folder_train'], sub_folders_train[config['split_train']:config['split_valid']]),
        'test': (config['folder_train'], sub_folders_train[config['split_valid']:config['split_test']]),
    }
    folders_all = {key: [os.path.join(val[0], sub_val) for sub_val in val[1]] for key, val in folders_all.items()}
    max_objects_all = {
        'train': config['max_objects_train'],
        'valid': config['max_objects_train'],
        'test': config['max_objects_train'],
    }
    generate_data = generate_data_multi if config['multiview'] else generate_data_single
    datasets = {}
    labels = {}
    class_map = None
    if config['add_class']:
        with open('./src/class_shop.json', 'r') as f:
            class_map = json.load(f)
    for phase, folders in folders_all.items():
        data_list = []
        label_list = []
        filename_all = os.listdir(folders[0])
        num_views = 0
        for filename in filename_all:
            if filename.split('_')[0] == 'image':
                num_views += 1
        config['current_num_views'] = num_views
        for folder in folders:
            print('processing {}'.format(folder))
            if config['multiview']:
                data, label = generate_data(config, folder, max_objects_all[phase], class_map=class_map)
                data_list.append(data)
                label_list.append(label)
            else:
                data = generate_data(config, folder, max_objects_all[phase], class_map=class_map)
                data_list.append(data)
        datasets[phase] = data_list
        if config['multiview']:
            labels[phase] = label_list
    save_dataset(config, datasets)
    if config['multiview']:
        config['name'] = label_name
        save_dataset(config, labels)
    return


if __name__ == '__main__':
    main()
