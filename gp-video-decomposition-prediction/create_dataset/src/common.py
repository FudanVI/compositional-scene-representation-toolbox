import os

import h5py
import numpy as np
import yaml
from skimage.transform import rescale, resize


def load_config(parser):
    args = parser.parse_args()
    with open(args.path_config) as f:
        config = yaml.safe_load(f)
    for key, val in args.__dict__.items():
        if key not in config or val is not None:
            config[key] = val
    if not os.path.exists(config['folder_out']):
        os.makedirs(config['folder_out'])
    if 'seed' in config:
        np.random.seed(config['seed'])
    return config


def crop_image(image):
    mask = image[..., -1] / 255
    valid_rows, = np.where(mask.sum(1) != 0)
    valid_cols, = np.where(mask.sum(0) != 0)
    row1, row2 = valid_rows.min(), valid_rows.max() + 1
    col1, col2 = valid_cols.min(), valid_cols.max() + 1
    image = image[row1:row2, col1:col2]
    return image


def rescale_image(image, scale):
    image = rescale(image, scale, mode='reflect', preserve_range=True, multichannel=True, anti_aliasing=True)
    return image.astype(np.uint8)


def resize_image(image, shape):
    image = resize(image, shape, mode='reflect', preserve_range=True, anti_aliasing=True)
    return image.astype(np.uint8)


def convert_layers(layers, th=0.5):
    image = layers[-1, ..., :-1]
    for layer in reversed(layers[:-1]):
        mask = layer[..., -1:] / 255
        image = image * (1 - mask) + layer[..., :-1] * mask
    image = image.astype(np.uint8)
    masks = layers[..., -1] / 255
    part_cumprod = np.concatenate([
        np.ones((1, *masks.shape[1:]), dtype=masks.dtype),
        np.cumprod(1 - masks[:-1], axis=0),
    ], axis=0)
    coefs = masks * part_cumprod
    segment = np.argmax(coefs, 0).astype(np.uint8)
    overlap = ((masks >= th).sum(0) - 1).astype(np.uint8)
    return image, segment, overlap


def save_dataset(config, datasets):
    with h5py.File(os.path.join(config['folder_out'], '{}.h5'.format(config['name'])), 'w') as f:
        for phase, data_list in datasets.items():
            data_groups = {key: [] for key in data_list[0]}
            for data in data_list:
                for key, val in data.items():
                    data_groups[key].append(val)
            f.create_group(phase)
            for key, val in data_groups.items():
                f[phase].create_dataset(key, data=np.stack(val), compression=True)
    return


def create_dataset(config, elements, generate_layers):
    datasets = {}
    for phase in ['train', 'valid', 'test', 'general']:
        phase_param = config[phase]
        if 'range_scale' in phase_param:
            assert len(phase_param['range_scale']) == 2
            assert phase_param['range_scale'][0] <= phase_param['range_scale'][1]
        data_list = []
        if phase in elements:
            sub_elements = elements[phase]
        else:
            sub_elements = elements
        for _ in range(phase_param['num_data']):
            layers = generate_layers(phase_param, sub_elements)
            image, segment, overlap = convert_layers(layers)
            data = {'image': image, 'segment': segment, 'overlap': overlap, 'layers': layers}
            data_list.append(data)
        datasets[phase] = data_list
    save_dataset(config, datasets)
    return
