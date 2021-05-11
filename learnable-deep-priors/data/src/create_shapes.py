import argparse
import numpy as np
from common import create_dataset


def generate_objects(elements, image_height, image_width, num_objects):
    objects = np.zeros((num_objects + 1, image_height, image_width, elements[0].shape[-1]), dtype=np.float32)
    objects[-1, ..., -1] = 1
    for offset in range(1, objects.shape[0]):
        idx = objects.shape[0] - offset - 1
        element = elements[np.random.randint(len(elements))]
        col1 = np.random.randint(image_width - element.shape[1] + 1)
        col2 = col1 + element.shape[1]
        row1 = np.random.randint(image_height - element.shape[0] + 1)
        row2 = row1 + element.shape[0]
        objects[idx, row1:row2, col1:col2] = element
    return objects


def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--name')
    parser.add_argument('--num_objects', type=int)
    parser.add_argument('--image_height', type=int)
    parser.add_argument('--image_width', type=int)
    parser.add_argument('--num_train', type=int, default=50000)
    parser.add_argument('--num_valid', type=int, default=10000)
    parser.add_argument('--num_test', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=265076)
    args = parser.parse_args()
    # Elements
    square = np.array(
        [[1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 0, 0, 0, 1, 1],
         [1, 1, 0, 0, 0, 1, 1],
         [1, 1, 0, 0, 0, 1, 1],
         [1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1]])
    triangle = np.array(
        [[0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
         [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
         [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    elements = [square, triangle, triangle[::-1, :].copy()]
    elements = [n[..., None].repeat(2, axis=-1) for n in elements]
    item_shape = [args.num_objects + 1, args.image_height, args.image_width, elements[0].shape[-1]]
    objects = {
        'train': np.empty((args.num_train, *item_shape), dtype=np.float32),
        'valid': np.empty((args.num_valid, *item_shape), dtype=np.float32),
        'test': np.empty((args.num_test, *item_shape), dtype=np.float32),
    }
    # Datasets
    np.random.seed(args.seed)
    for key in ['train', 'valid', 'test']:
        for idx in range(objects[key].shape[0]):
            objects[key][idx] = generate_objects(elements, args.image_height, args.image_width, args.num_objects)
    create_dataset(args.name, objects)
    return


if __name__ == '__main__':
    main()
