import argparse
import numpy as np
import os
from imageio import imread
from multiprocessing.pool import Pool
from skimage.color import rgb2hsv, hsv2rgb
from common import load_config, crop_image, rescale_image, resize_image, create_dataset


def augment_hsv(x):
    out_list = []
    for h_offset in np.linspace(-0.1, 0.1, 32):
        for s_scale in np.linspace(0.9, 1, 16):
            for v_scale in np.linspace(0.9, 1, 16):
                out_list.append((x, h_offset, s_scale, v_scale))
    return out_list


def convert_image(data):
    image, h_offset, s_scale, v_scale = data
    image_rgb, image_a = image[..., :-1], image[..., -1:]
    image_hsv = rgb2hsv(image_rgb)
    image_h_new = (image_hsv[..., 0] + h_offset) % 1.0
    image_s_new = image_hsv[..., 1] * s_scale
    image_v_new = image_hsv[..., 2] * v_scale
    image_hsv_new = np.stack([image_h_new, image_s_new, image_v_new], axis=-1)
    image_rgb_new = (hsv2rgb(image_hsv_new) * 255).astype(np.uint8)
    image_new = np.concatenate([image_rgb_new, image_a], axis=-1)
    return image_new


def convert_abstract(config):
    back = imread(os.path.join(config['folder_in'], '{}.png'.format(config['back'])))
    back = resize_image(back, (config['image_height'], config['image_width']))
    back[..., -1] = 255
    back_list = augment_hsv(back)
    with Pool(config['num_parallel']) as pool:
        back_list = pool.map(convert_image, back_list)
    objects = []
    for item in config['objects']:
        image = imread(os.path.join(config['folder_in'], '{}.png'.format(item['name'])))
        image = crop_image(image)
        image = rescale_image(image, item['scale'])
        image_list = augment_hsv(image)
        with Pool(config['num_parallel']) as pool:
            image_list = pool.map(convert_image, image_list)
        obj_dict = {'image': image_list, 'type': item['type']}
        for key in ['row_min', 'row_max']:
            if key in item:
                obj_dict[key] = round(config['image_height'] * item[key])
        objects.append(obj_dict)
    elements = {'back': back_list, 'objects': objects}
    return elements


def generate_layers(config, elements, max_tries=10):
    idx_sel = np.random.randint(len(elements['back']))
    back = elements['back'][idx_sel]
    max_objects = max(config['num_objects'])
    num_objects = np.random.choice(config['num_objects'])
    while True:
        layers = np.zeros((max_objects, *back.shape), dtype=np.uint8)
        masks = np.zeros((num_objects, *back.shape[:-1]), dtype=np.float64)
        index = np.random.choice(len(elements['objects']), size=num_objects, replace=False)
        objects = [elements['objects'][idx] for idx in index]
        objects = sorted(objects, key=lambda x: x['type'], reverse=True)
        row_list = []
        for idx, obj_dict in enumerate(objects):
            idx_sel = np.random.randint(len(obj_dict['image']))
            image_base = obj_dict['image'][idx_sel]
            for _ in range(max_tries):
                if obj_dict['type'] == 2:
                    row2 = np.random.randint(obj_dict['row_min'], back.shape[0] + 1)
                    ratio = (back.shape[0] - row2) / (back.shape[0] - obj_dict['row_min'])
                    scale = ratio * config['range_scale'][0] + (1 - ratio) * config['range_scale'][1]
                    image = rescale_image(image_base, scale)
                    row1 = row2 - image.shape[0]
                    if row1 < 0:
                        continue
                    row_list.append(row2)
                else:
                    ratio = np.random.rand()
                    scale = ratio * config['range_scale'][0] + (1 - ratio) * config['range_scale'][1]
                    image = rescale_image(image_base, scale)
                    if image.shape[1] > obj_dict['row_max']:
                        continue
                    row2 = np.random.randint(image.shape[0], obj_dict['row_max'] + 1)
                    row1 = row2 - image.shape[0]
                col1 = np.random.randint(back.shape[1] - image.shape[1] + 1)
                col2 = col1 + image.shape[1]
                layers[idx, row1:row2, col1:col2] = image
                masks[idx, row1:row2, col1:col2] = image[..., -1] / 255
                for sub_idx in range(idx + 1):
                    mask = masks[sub_idx]
                    visible = (1 - masks[:sub_idx]).prod(0) * (1 - masks[sub_idx + 1:idx + 1]).prod(0)
                    if (mask * visible).sum() < config['min_visible'] * mask.sum():
                        layers[idx, row1:row2, col1:col2] = 0
                        masks[idx, row1:row2, col1:col2] = 0
                        if obj_dict['type'] == 2:
                            row_list.pop()
                        break
                else:
                    break
            else:
                break
        else:
            break
    if len(row_list) > 0:
        layers_part1 = layers[:len(row_list)]
        layers_part2 = layers[len(row_list):num_objects]
        layers_part3 = layers[num_objects:]
        index = np.argsort(row_list)
        layers_part1 = np.stack([layers_part1[idx] for idx in reversed(index)])
        layers = np.concatenate([layers_part1, layers_part2, layers_part3])
    layers = np.concatenate([layers, back[None]])
    return layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name')
    parser.add_argument('--path_config')
    parser.add_argument('--folder_in')
    parser.add_argument('--folder_out')
    parser.add_argument('--num_parallel', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    config = load_config(parser)
    elements = convert_abstract(config)
    create_dataset(config, elements, generate_layers)
    return


if __name__ == '__main__':
    main()
