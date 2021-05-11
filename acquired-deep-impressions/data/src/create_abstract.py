import argparse
import numpy as np
import os
import re
from imageio import imread
from skimage.transform import resize
from common import load_config, crop_image, rescale_image, create_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config')
    parser.add_argument('--folder_in')
    parser.add_argument('--folder_out')
    parser.add_argument('--seed', type=int, default=0)
    config = load_config(parser)
    back = imread(os.path.join(config['folder_in'], 'background.png'))
    back_shape = (config['image_height'], config['image_width'])
    back = resize(back, back_shape, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    back[..., -1] = 255
    back_list = [back]
    pattern = re.compile(r'^.*?_\d*?s\.png')
    filenames_all = [n for n in sorted(os.listdir(config['folder_in'])) if pattern.match(n)]
    pattern = re.compile(r'^hb\d_.*?')
    filenames_single = [n for n in filenames_all if pattern.match(n)]
    filenames_rest = [n for n in filenames_all if n not in filenames_single]
    filenames = filenames_single + filenames_rest
    objects_list = []
    for cls, filename in enumerate(filenames):
        image = imread(os.path.join(config['folder_in'], filename))
        image = crop_image(image, image[..., -1])
        ratio_row = config['image_height'] / image.shape[0] / config['range_scale'][1]
        ratio_col = config['image_width'] / image.shape[1] / config['range_scale'][1]
        scale = min(ratio_row, ratio_col, np.sqrt(0.25 * ratio_row * ratio_col))
        image = rescale_image(image, scale)
        objects_list.append([image])
    create_dataset(config, back_list, objects_list, name='abstract')
    return


if __name__ == '__main__':
    main()
