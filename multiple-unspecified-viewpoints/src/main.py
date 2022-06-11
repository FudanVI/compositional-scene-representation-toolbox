import argparse
import datetime
import numpy as np
import os
import random
import torch
import yaml
from dataset import get_data_loaders
from model import get_model
from run_model import train_model, test_model, test_model_multi, test_model_cond, test_model_latent


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config')
    parser.add_argument('--path_data')
    parser.add_argument('--path_pretrain')
    parser.add_argument('--folder_log')
    parser.add_argument('--folder_out')
    parser.add_argument('--timestamp')
    parser.add_argument('--num_tests', type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--without_oc', action='store_true')
    parser.add_argument('--use_timestamp', action='store_true')
    parser.add_argument('--file_ckpt', default='ckpt.pth')
    parser.add_argument('--file_model', default='model.pth')
    args = parser.parse_args()
    with open(args.path_config) as f:
        config = yaml.safe_load(f)
    for key, val in args.__dict__.items():
        if key not in config or val is not None:
            config[key] = val
    if config['debug']:
        config['ckpt_intvl'] = 1
    if config['resume']:
        config['train'] = True
    if config['timestamp'] is None:
        config['timestamp'] = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    if config['use_timestamp']:
        for key in ['folder_log', 'folder_out']:
            config[key] = os.path.join(config[key], config['timestamp'])
    if 'seed' not in config:
        config['seed'] = random.randint(0, 0xffffffff)
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return


def main():
    config = get_config()
    if config['train'] and not config['resume']:
        for key in ['folder_log', 'folder_out']:
            if os.path.exists(config[key]):
                raise FileExistsError(config[key])
            os.makedirs(config[key])
        with open(os.path.join(config['folder_out'], 'config.yaml'), 'w') as f:
            yaml.safe_dump(config, f)
    set_seed(config['seed'])
    data_loaders, image_shape = get_data_loaders(config)
    config['image_shape'] = image_shape
    set_seed(config['seed'])
    net = get_model(config)
    if config['train']:
        set_seed(config['seed'])
        train_model(config, data_loaders, net)
    set_seed(config['seed'])
    if config['phase_param']['train']['num_views'] == 1:
        test_model(config, data_loaders, net)
    else:
        test_model_multi(config, data_loaders, net)
        if not config['without_oc']:
            test_model_cond(config, data_loaders, net)
            test_model_latent(config, data_loaders, net)
    return


if __name__ == '__main__':
    main()
