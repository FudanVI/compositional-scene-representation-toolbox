import argparse
import numpy as np
import torchvision
from common import load_config, crop_image, create_dataset


def convert_mnist(image):
    image = crop_image(image, image)
    value = np.full_like(image, image.max())
    mask = ((image / value) * 255).astype(value.dtype)
    return np.stack([value, mask], axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config')
    parser.add_argument('--folder_downloads')
    parser.add_argument('--folder_out')
    parser.add_argument('--seed', type=int, default=0)
    config = load_config(parser)
    back = np.zeros((config['image_height'], config['image_width'], 2), dtype=np.uint8)
    back[..., -1] = 255
    back_list = [back]
    mnist = [(convert_mnist(np.array(image)), cls) for train in [True, False]
             for (image, cls) in torchvision.datasets.MNIST(config['folder_downloads'], train=train, download=True)]
    num_classes = 10
    objects_list = [[] for _ in range(num_classes)]
    for image, cls in mnist:
        objects_list[cls].append(image)
    cnt = 0
    for n in objects_list[:5]:
        cnt += len(n)
    create_dataset(config, back_list, objects_list, name='mnist')
    return


if __name__ == '__main__':
    main()
