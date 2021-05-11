import argparse
import numpy as np
import os
import torchvision
from common import generate_objects, create_dataset


def create_element(image):
    image = image[image.sum(1) != 0][:, image.sum(0) != 0] / 255
    value = np.full_like(image, image.max())
    mask = image / value
    element = np.stack([value, mask], axis=-1)
    return element


def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--name')
    parser.add_argument('--folder_outputs')
    parser.add_argument('--occlusion', type=int)
    parser.add_argument('--num_objects_all', type=int, nargs='+')
    parser.add_argument('--folder_downloads', default='downloads')
    parser.add_argument('--num_train', type=int, default=50000)
    parser.add_argument('--num_valid', type=int, default=10000)
    parser.add_argument('--num_test', type=int, default=10000)
    parser.add_argument('--image_height', type=int, default=48)
    parser.add_argument('--image_width', type=int, default=48)
    parser.add_argument('--seed', type=int, default=265076)
    args = parser.parse_args()
    if not os.path.exists(args.folder_outputs):
        os.mkdir(args.folder_outputs)
    # Elements
    mnist = {
        phase: torchvision.datasets.MNIST(args.folder_downloads, train=train)
        for phase, train in zip(['train', 'test'], [True, False])
    }
    elements = {key: [create_element(np.array(n[0])) for n in val] for key, val in mnist.items()}
    back = np.zeros((args.image_height, args.image_width, 2))
    back[..., -1] = 1
    elements = {key: {'back': back, 'objects': elements[key_prev]}
                for key, key_prev in zip(['train', 'valid', 'test'], ['train', 'train', 'test'])}
    # Objects
    objects = generate_objects(args, elements)
    create_dataset(os.path.join(args.folder_outputs, args.name), objects)
    return


if __name__ == '__main__':
    main()
