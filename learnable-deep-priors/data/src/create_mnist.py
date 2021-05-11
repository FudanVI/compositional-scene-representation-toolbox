import argparse
import numpy as np
import os
import torchvision
from common import create_dataset


def create_element(image):
    image = image[image.sum(1) != 0][:, image.sum(0) != 0] / 255
    value = np.full_like(image, image.max())
    mask = image / value
    element = np.stack([value, mask], axis=-1)
    return element


def generate_objects(elements, image_height, image_width, num_objects):
    objects = np.zeros((num_objects + 1, image_height, image_width, elements[0].shape[-1]), dtype=np.float32)
    objects[-1, ..., -1] = 1
    indices = np.random.randint(len(elements), size=num_objects)
    for offset in range(1, objects.shape[0]):
        idx = objects.shape[0] - offset - 1
        element = elements[indices[offset - 1]]
        col1 = np.random.randint(image_width - element.shape[1] + 1)
        col2 = col1 + element.shape[1]
        row1 = np.random.randint(image_height - element.shape[0] + 1)
        row2 = row1 + element.shape[0]
        objects[idx, row1:row2, col1:col2] = element
    return objects


def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='mnist')
    parser.add_argument('--folder_downloads', default='downloads')
    parser.add_argument('--num_objects', type=int, default=2)
    parser.add_argument('--image_height', type=int, default=48)
    parser.add_argument('--image_width', type=int, default=48)
    parser.add_argument('--num_train', type=int, default=50000)
    parser.add_argument('--num_valid', type=int, default=10000)
    parser.add_argument('--num_test', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=36520)
    args = parser.parse_args()
    # Elements
    if not os.path.exists(args.folder_downloads):
        os.mkdir(args.folder_downloads)
    mnist = {
        phase: torchvision.datasets.MNIST(args.folder_downloads, train=train, download=True)
        for phase, train in zip(['train', 'test'], [True, False])
    }
    elements = {key: [create_element(np.array(n[0])) for n in val] for key, val in mnist.items()}
    item_shape = [args.num_objects + 1, args.image_height, args.image_width, elements['train'][0].shape[-1]]
    objects = {
        'train': np.empty((args.num_train, *item_shape), dtype=np.float32),
        'valid': np.empty((args.num_valid, *item_shape), dtype=np.float32),
        'test': np.empty((args.num_test, *item_shape), dtype=np.float32),
    }
    # Datasets (20 and 500)
    for num_variants in [20, 500]:
        np.random.seed(args.seed)
        for key in ['train', 'valid', 'test']:
            for idx in range(objects[key].shape[0]):
                objects[key][idx] = generate_objects(
                    elements['train'][:num_variants], args.image_height, args.image_width, args.num_objects)
        create_dataset('{}_{}'.format(args.name, num_variants), objects)
    # Datasets (all)
    np.random.seed(args.seed)
    for key in ['train', 'valid']:
        for idx in range(objects[key].shape[0]):
            objects[key][idx] = generate_objects(
                elements['train'], args.image_height, args.image_width, args.num_objects)
    key = 'test'
    for idx in range(objects[key].shape[0]):
        objects[key][idx] = generate_objects(elements['test'], args.image_height, args.image_width, args.num_objects)
    create_dataset('{}_all'.format(args.name), objects)
    return


if __name__ == '__main__':
    main()
