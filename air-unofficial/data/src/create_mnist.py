import argparse
import h5py
import numpy as np
import os
import torchvision
from skimage.transform import rescale


def create_element(image):
    image = image / 255
    value = np.full_like(image, image.max())
    mask = image / value
    element = np.stack([value, mask], axis=-1)
    return element


def generate_objects_sub(elements, num_objects, max_objects):
    objects = np.zeros((max_objects + 1, *elements['back'].shape), dtype=np.float32)
    while True:
        objects.fill(0)
        objects[-1] = elements['back']
        canvas = elements['back'][..., :-1]
        class_list = []
        for idx in range(num_objects):
            scale = 1 / (1.3 + 0.1 * np.random.randn())
            sub_idx = np.random.randint(len(elements['objects']))
            image = rescale(elements['objects'][sub_idx], scale, preserve_range=True, multichannel=True)
            row1 = np.random.randint(elements['back'].shape[0] - image.shape[0] + 1)
            row2 = row1 + image.shape[0]
            col1 = np.random.randint(elements['back'].shape[1] - image.shape[1] + 1)
            col2 = col1 + image.shape[1]
            objects[idx, row1:row2, col1:col2] = image
            canvas = canvas + objects[idx, ..., :-1] * objects[idx, ..., -1:]
            if canvas.max() > 1:
                break
            class_list.append(elements['classes'][sub_idx])
        else:
            break
    class_list += [255] * (max_objects - num_objects + 1)
    return objects, np.array(class_list)


def generate_objects(args, elements):
    num_data = {'train': args.num_train, 'test': args.num_test}
    max_objects = max(args.num_objects_list)
    objects = {key: np.zeros((num_data[key], max_objects + 1, *elements[key]['back'].shape),
                             dtype=np.float32) for key in num_data}
    classes = {key: np.full((num_data[key], max_objects + 1), 255, dtype=np.uint8) for key in num_data}
    np.random.seed(args.seed)
    for key, val in elements.items():
        num_objects_list = np.random.choice(args.num_objects_list, size=num_data[key])
        for idx, num_objects in enumerate(num_objects_list):
            objects[key][idx], classes[key][idx] = generate_objects_sub(val, num_objects, max_objects)
    return objects, classes


def generate_images(objects):
    images = (objects.prod(-1, keepdims=True).sum(1) * 255).astype(np.uint8)
    return images


def generate_labels(objects, th=0.5):
    masks = objects[..., -1]
    part_cumprod = np.concatenate([
        np.ones((masks.shape[0], 1, *masks.shape[2:]), dtype=masks.dtype),
        np.cumprod(1 - masks[:, :-1], axis=1),
    ], axis=1)
    coef = masks * part_cumprod
    segments = np.argmax(coef, axis=1).astype(np.uint8)
    overlaps = ((masks >= th).sum(1) - 1).astype(np.uint8)
    labels = {'segment': segments, 'overlap': overlaps}
    return labels


def save_dataset(name, images, labels, objects, classes):
    with h5py.File('{}.h5'.format(name), 'w') as f:
        for key in images:
            f.create_group(key)
            f[key].create_dataset('image', data=images[key], compression='gzip')
            f[key].create_dataset('segment', data=labels[key]['segment'], compression='gzip')
            f[key].create_dataset('overlap', data=labels[key]['overlap'], compression='gzip')
            f[key].create_dataset('layers', data=objects[key], compression='gzip')
            f[key].create_dataset('classes', data=classes[key], compression='gzip')
    return


def create_dataset(name, objects, classes):
    images = {key: generate_images(val) for key, val in objects.items()}
    labels = {key: generate_labels(val) for key, val in objects.items()}
    objects = {key: (val * 255).astype(np.uint8) for key, val in objects.items()}
    save_dataset(name, images, labels, objects, classes)
    return


def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--name')
    parser.add_argument('--num_objects_list', type=int, nargs='+')
    parser.add_argument('--folder_downloads', default='downloads')
    parser.add_argument('--min_visible', type=float, default=1.0)
    parser.add_argument('--num_train', type=int, default=60000)
    parser.add_argument('--num_test', type=int, default=10000)
    parser.add_argument('--image_height', type=int, default=50)
    parser.add_argument('--image_width', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if not os.path.exists(args.folder_downloads):
        os.mkdir(args.folder_downloads)
    # Elements
    mnist = torchvision.datasets.MNIST(args.folder_downloads, train=True, download=True)
    elements = [create_element(np.array(n[0])) for n in mnist]
    classes = [n[1] for n in mnist]
    back = np.zeros((args.image_height, args.image_width, 2))
    back[..., -1] = 1
    elements = {key: {'back': back, 'objects': elements, 'classes': classes} for key in ['train', 'test']}
    # Objects
    objects, classes = generate_objects(args, elements)
    create_dataset(args.name, objects, classes)


if __name__ == '__main__':
    main()
