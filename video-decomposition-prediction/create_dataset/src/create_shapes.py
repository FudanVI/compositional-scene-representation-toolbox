import argparse
import numpy as np
from common import load_config, rescale_image, create_dataset


def generate_layers(config, elements, max_tries=10):
    back_shape = elements['back'].shape
    max_objects = max(config['num_objects'])
    num_objects = np.random.choice(config['num_objects'])
    while True:
        layers = np.zeros((max_objects, *back_shape), dtype=np.uint8)
        masks = np.zeros((num_objects, *back_shape[:-1]), dtype=np.float64)
        index = np.random.randint(len(elements['objects']), size=num_objects)
        objects = [elements['objects'][idx] for idx in index]
        for idx, image_base in enumerate(objects):
            ratio = np.random.rand()
            scale = ratio * config['range_scale'][0] + (1 - ratio) * config['range_scale'][1]
            image = rescale_image(image_base, scale)
            image[..., :-1] = 255
            for _ in range(max_tries):
                row1 = np.random.randint(back_shape[0] - image.shape[0] + 1)
                row2 = row1 + image.shape[0]
                col1 = np.random.randint(back_shape[1] - image.shape[1] + 1)
                col2 = col1 + image.shape[1]
                layers[idx, row1:row2, col1:col2] = image
                masks[idx, row1:row2, col1:col2] = image[..., -1] / 255
                for sub_idx in range(idx + 1):
                    mask = masks[sub_idx]
                    visible = (1 - masks[:sub_idx]).prod(0) * (1 - masks[sub_idx + 1:idx + 1]).prod(0)
                    if (mask * visible).sum() < config['min_visible'] * mask.sum():
                        layers[idx, row1:row2, col1:col2] = 0
                        masks[idx, row1:row2, col1:col2] = 0
                        break
                else:
                    break
            else:
                break
        else:
            break
    layers = np.concatenate([layers, elements['back'][None]])
    return layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name')
    parser.add_argument('--path_config')
    parser.add_argument('--folder_out')
    parser.add_argument('--seed', type=int, default=0)
    config = load_config(parser)
    back = np.zeros((config['image_height'], config['image_width'], 2), dtype=np.uint8)
    back[..., -1] = 255
    square = np.array(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
         [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
         [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
         [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
         [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
         [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8)
    triangle = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8)
    objects = [square, triangle, triangle[::-1].copy()]
    objects = [n[..., None].repeat(2, axis=-1) * 255 for n in objects]
    elements = {'back': back, 'objects': objects}
    create_dataset(config, elements, generate_layers)
    return


if __name__ == '__main__':
    main()
