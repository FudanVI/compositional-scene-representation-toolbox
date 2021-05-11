import h5py
import numpy as np
import os
import yaml
from skimage.transform import rescale
from skimage.color import rgb2yuv, yuv2rgb


def load_config(parser):
    args = parser.parse_args()
    with open(args.path_config) as f:
        config = yaml.safe_load(f)
    for key, val in args.__dict__.items():
        if key not in config or val is not None:
            config[key] = val
    assert len(config['range_scale']) == 2
    assert config['range_scale'][0] <= config['range_scale'][1]
    if 'range_intensity' in config:
        assert len(config['range_intensity']) == 2
        assert config['range_intensity'][0] <= config['range_intensity'][1]
    for idx in range(len(config['sep_occlude_list']) - 1):
        assert config['sep_occlude_list'][idx] < config['sep_occlude_list'][idx + 1]
    if not os.path.exists(config['folder_out']):
        os.makedirs(config['folder_out'])
    np.random.seed(config['seed'])
    return config


def crop_image(image, mask):
    valid_rows, = np.where(mask.sum(1) != 0)
    valid_cols, = np.where(mask.sum(0) != 0)
    row1, row2 = valid_rows.min(), valid_rows.max() + 1
    col1, col2 = valid_cols.min(), valid_cols.max() + 1
    image = image[row1:row2, col1:col2]
    return image


def rescale_image(image, scale):
    return rescale(image, scale, preserve_range=True, multichannel=True, anti_aliasing=True).astype(np.uint8)


def generate_layers_simple(config, back_list, objects_all, num_objects_list):
    max_objects = max(num_objects_list)
    num_objects = np.random.choice(num_objects_list)
    idx_back = np.random.randint(0, len(back_list))
    back = back_list[idx_back]
    image_shape = back.shape
    obj_layers = np.zeros((max_objects, *image_shape), dtype=np.uint8)
    occ_masks = np.zeros((num_objects, *image_shape[:-1]), dtype=np.float)
    indices = np.random.randint(len(objects_all), size=num_objects)
    scales = np.random.uniform(config['range_scale'][0], config['range_scale'][1], size=num_objects)
    obj_images = [rescale_image(objects_all[idx][0], scale) for idx, scale in zip(indices, scales)]
    if 'range_intensity' in config:
        y_scales = np.random.uniform(config['range_intensity'][0], config['range_intensity'][1], size=num_objects)
        yuv_images = [rgb2yuv(image[..., :-1]) for image in obj_images]
        for image, y_scale in zip(yuv_images, y_scales):
            image[..., 0] *= y_scale
        rgb_images = [(np.clip(yuv2rgb(image), 0, 1) * 255).astype(np.uint8) for image in yuv_images]
        obj_images = [np.concatenate([rgb, obj[..., -1:]], axis=-1) for rgb, obj in zip(rgb_images, obj_images)]
    obj_classes = np.array([objects_all[idx][1] for idx in indices])
    for idx, image in enumerate(obj_images):
        row1 = np.random.randint(image_shape[0] - image.shape[0] + 1)
        row2 = row1 + image.shape[0]
        col1 = np.random.randint(image_shape[1] - image.shape[1] + 1)
        col2 = col1 + image.shape[1]
        obj_layers[idx, row1:row2, col1:col2] = image
        occ_masks[idx, row1:row2, col1:col2] = 1 if config['overlap_bbox'] else image[..., -1] / 255
    layers = np.concatenate([back[None], obj_layers])
    layers = np.rollaxis(layers, -1, -3)
    classes = np.full(max_objects + 1, -1, dtype=np.int8)
    classes[1:obj_classes.shape[0] + 1] = obj_classes
    return layers, classes, occ_masks


def compute_occlude_index(occ_masks, sep_occlude_list):
    occ_ratio_list = []
    for idx, mask in enumerate(occ_masks):
        region_exclude = 1 - (1 - occ_masks[:idx]).prod(0) * (1 - occ_masks[idx + 1:]).prod(0)
        occ_ratio = (region_exclude * mask).sum() / mask.sum()
        occ_ratio_list.append(occ_ratio)
    avg_occ_ratio = np.mean(occ_ratio_list)
    idx_occ = None
    for idx in range(len(sep_occlude_list) - 1):
        if sep_occlude_list[idx] <= avg_occ_ratio < sep_occlude_list[idx + 1]:
            idx_occ = idx
            break
    return idx_occ


def convert_layers(layers, th=0.5):
    image = layers[0, :-1]
    for layer in layers[1:]:
        mask = layer[-1:] / 255
        image = image * (1 - mask) + layer[:-1] * mask
    image = image.astype(np.uint8)
    masks_rev = layers[::-1, -1] / 255
    part_cumprod = np.concatenate([np.ones((1, *masks_rev.shape[1:]), dtype=masks_rev.dtype),
                                   np.cumprod(1 - masks_rev[:-1], axis=0)], axis=0)
    coefs = (masks_rev * part_cumprod)[::-1]
    segment = np.argmax(coefs, 0).astype(np.uint8)
    overlap = ((masks_rev >= th).sum(0) - 1).astype(np.uint8)
    return image, segment, overlap


def create_dataset(config, back_list, objects_list, name, generate_layers_fn=generate_layers_simple):
    objects_sel = [(image, cls) for cls in range(config['single_classes']) for image in objects_list[cls]]
    data_list = []
    for _ in range(config['single_data']):
        layers, classes, _ = generate_layers_fn(config, back_list, objects_sel, num_objects_list=[1])
        image, segment, overlap = convert_layers(layers)
        data = {'image': image, 'segment': segment, 'overlap': overlap, 'layers': layers, 'classes': classes}
        data_list.append(data)
    datasets = {'single': data_list}
    for phase in ['train', 'valid', 'test', 'general']:
        objects_sel = [(image, cls) for cls, images in enumerate(objects_list) for image in images]
        sub_config = config[phase]
        data_list_all = [[] for _ in range(len(config['sep_occlude_list']) - 1)]
        while min([len(data_list) for data_list in data_list_all]) < sub_config['num_data']:
            layers, classes, occ_masks = generate_layers_fn(config, back_list, objects_sel, sub_config['num_objects_list'])
            idx_occ = compute_occlude_index(occ_masks, config['sep_occlude_list'])
            if idx_occ is not None and len(data_list_all[idx_occ]) < sub_config['num_data']:
                image, segment, overlap = convert_layers(layers)
                data = {'image': image, 'segment': segment, 'overlap': overlap, 'layers': layers, 'classes': classes}
                data_list_all[idx_occ].append(data)
        for idx, data_list in enumerate(data_list_all):
            datasets['{}_{}'.format(phase, idx)] = data_list
    with h5py.File(os.path.join(config['folder_out'], '{}.h5'.format(name)), 'w') as f:
        for phase, data_list in datasets.items():
            data_groups = {key: [] for key in data_list[0]}
            for data in data_list:
                for key, val in data.items():
                    data_groups[key].append(val)
            f.create_group(phase)
            for key, val in data_groups.items():
                f[phase].create_dataset(key, data=np.stack(val), compression=True)
    return
