import h5py
import numpy as np


def generate_images(objects):
    images = objects[:, -1, ..., :-1]
    for offset in range(1, objects.shape[1]):
        idx = objects.shape[1] - offset - 1
        masks = objects[:, idx, ..., -1:]
        images = images * (1 - masks) + objects[:, idx, ..., :-1] * masks
    images = (images * 255).astype(np.uint8)
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


def save_dataset(name, images, labels, objects):
    with h5py.File('{}.h5'.format(name), 'w') as f:
        for key in images:
            f.create_group(key)
            f[key].create_dataset('image', data=images[key], compression='gzip')
            f[key].create_dataset('segment', data=labels[key]['segment'], compression='gzip')
            f[key].create_dataset('overlap', data=labels[key]['overlap'], compression='gzip')
            f[key].create_dataset('layers', data=objects[key], compression='gzip')
    return


def create_dataset(name, objects):
    images = {key: generate_images(val) for key, val in objects.items()}
    labels = {key: generate_labels(val) for key, val in objects.items()}
    objects = {key: (val * 255).astype(np.uint8) for key, val in objects.items()}
    save_dataset(name, images, labels, objects)
    return
