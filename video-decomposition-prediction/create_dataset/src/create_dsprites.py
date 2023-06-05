import argparse
import numpy as np
from multiprocessing.pool import Pool
from common import load_config, create_dataset


def convert_dsprites(x):
    return np.stack([np.full_like(x, 255), x], axis=-1)


def generate_layers(config, elements, max_tries=10):
    back_shape = elements['back'].shape
    max_objects = max(config['num_objects'])
    num_objects = np.random.choice(config['num_objects'])
    while True:
        layers = np.zeros((max_objects, *back_shape), dtype=np.uint8)
        masks = np.zeros((num_objects, *back_shape[:-1]), dtype=np.float64)
        index = np.random.randint(len(elements['objects']), size=num_objects)
        objects = [elements['objects'][idx] for idx in index]
        for idx, image in enumerate(objects):
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
    parser.add_argument('--path_in')
    parser.add_argument('--folder_out')
    parser.add_argument('--num_parallel', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    config = load_config(parser)
    back = np.zeros((config['image_height'], config['image_width'], 2), dtype=np.uint8)
    back[..., -1] = 255
    dsprites = np.load(config['path_in'])['imgs'] * 255
    assert back.shape[:2] == dsprites.shape[-2:]
    with Pool(config['num_parallel']) as pool:
        objects = pool.map(convert_dsprites, dsprites)
    elements = {'back': back, 'objects': objects}
    create_dataset(config, elements, generate_layers)
    return


if __name__ == '__main__':
    main()
