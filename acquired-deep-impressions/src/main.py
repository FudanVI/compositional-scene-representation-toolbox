import argparse
import datetime
import os
import yaml
from dataset import get_data_loaders
from model import get_model
from run_model import train_model, test_model


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config')
    parser.add_argument('--path_data')
    parser.add_argument('--path_pretrain')
    parser.add_argument('--folder_log')
    parser.add_argument('--folder_out')
    parser.add_argument('--timestamp')
    parser.add_argument('--split_train', type=int)
    parser.add_argument('--split_test', type=int)
    parser.add_argument('--num_tests', type=int)
    parser.add_argument('--coef_kld', type=float)
    parser.add_argument('--coef_mask', type=float)
    parser.add_argument('--mode_single', action='store_true')
    parser.add_argument('--seq_update', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--use_timestamp', action='store_true')
    parser.add_argument('--save_detail', action='store_true')
    parser.add_argument('--seg_bck', action='store_true')
    parser.add_argument('--file_ckpt', default='ckpt.pickle')
    parser.add_argument('--file_model', default='model.pickle')
    args = parser.parse_args()
    with open(args.path_config) as f:
        config = yaml.safe_load(f)
    for key, val in args.__dict__.items():
        if key not in config or val is not None:
            config[key] = val
    if config['resume']:
        config['train'] = True
    if config['timestamp'] is None:
        config['timestamp'] = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    if config['use_timestamp']:
        for key in ['folder_log', 'folder_out']:
            config[key] = os.path.join(config[key], config['timestamp'])
    return config


def main():
    config = get_config()
    if config['train'] and not config['resume']:
        for key in ['folder_log', 'folder_out']:
            if os.path.exists(config[key]):
                raise FileExistsError(config[key])
            os.makedirs(config[key])
        with open(os.path.join(config['folder_out'], 'config.yaml'), 'w') as f:
            yaml.safe_dump(config, f)
    data_loaders, image_shape = get_data_loaders(config)
    config['image_shape'] = image_shape
    if 'crop_shape' not in config:
        config['crop_shape'] = [val if idx == 0 else val // 2 for idx, val in enumerate(image_shape)]
    net = get_model(config)
    net_gen = None if config['path_pretrain'] is None else get_model(config)
    if config['train']:
        train_model(config, data_loaders, net, net_gen)
    test_model(config, data_loaders, net)
    return


if __name__ == '__main__':
    main()
