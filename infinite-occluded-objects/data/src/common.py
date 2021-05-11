import h5py
import numpy as np


def generate_objects_sub(args, elements, max_objects, num_objects, max_tries=10):
    back = elements['back'].astype(np.float32)
    objects = np.zeros((max_objects, *back.shape), dtype=np.float32)
    bboxs = np.ones(elements['back'].shape[:-1], dtype=np.bool)
    while True:
        objects.fill(0)
        bboxs.fill(True)
        sub_elements = []
        indices = np.random.randint(len(elements['objects']), size=num_objects)
        for idx in indices:
            sub_elements.append(elements['objects'][idx])
        for offset in range(num_objects):
            idx = num_objects - offset - 1
            image = sub_elements[offset]
            valid = False
            for _ in range(max_tries):
                row1 = np.random.randint(elements['back'].shape[0] - image.shape[0] + 1)
                row2 = row1 + image.shape[0]
                col1 = np.random.randint(elements['back'].shape[1] - image.shape[1] + 1)
                col2 = col1 + image.shape[1]
                if args.occlusion or np.all(bboxs[row1:row2, col1:col2]):
                    objects[idx, row1:row2, col1:col2] = image
                    bboxs[row1:row2, col1:col2] = False
                    valid = True
                    break
            if not valid:
                break
        else:
            break
    objects = np.concatenate([objects, back[None]])
    return objects


def generate_objects(args, elements):
    num_data = {'train': args.num_train, 'valid': args.num_valid, 'test': args.num_test}
    num_objects_all = np.array(args.num_objects_all)
    max_objects = num_objects_all.max()
    objects = {key: np.zeros((num_data[key], max_objects + 1, *elements[key]['back'].shape), dtype=np.float32)
               for key in num_data}
    np.random.seed(args.seed)
    for key, val in elements.items():
        num_objects_list = np.random.choice(num_objects_all, size=num_data[key])
        for idx, num_objects in enumerate(num_objects_list):
            objects[key][idx] = generate_objects_sub(args, val, max_objects, num_objects)
    return objects


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
