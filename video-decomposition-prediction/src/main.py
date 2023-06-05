import argparse
import datetime
import os
import pdb
import random

import numpy as np
import torch
import yaml

from dataset import get_data_loader
from model import get_model
from run_model import train_model, test_model_statistics, test_model_latent,test_model_observe,test_model_predict


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return


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
    parser.add_argument('--use_timestamp', action='store_true')
    parser.add_argument('--file_ckpt', default='ckpt.pth')
    parser.add_argument('--file_model', default='model.pth')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    # args.path_data = '/home/ctl/conference/gcm/dataset/clevr_multi/clevr_multi_simple_10.h5'
    # args.folder_log = '../../logs/gp_two_stage_transformer/clevr_multi_simple_10/debug'
    # args.folder_out = '../../outs/gp_two_stage_transformer/clevr_multi_simple_10/debug'
    # args.path_config = '../experiments/clevr_simple/config_first_stage.yaml'
    # args.train = True
    # args.resume = True
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
    if config['seed'] is None:
        config['seed'] = random.randint(0, 0xffffffff)
    print('seed = {}'.format(config['seed']))
    return config


def main():
    config = get_config()
    if config['train'] and not config['resume']:
        for key in ['folder_log', 'folder_out']:
            if os.path.exists(config[key]):
                print('{} has been created'.format(config[key]))
            else:
                print('creating {}'.format(config[key]))
                os.makedirs(config[key])
    if not os.path.exists(os.path.join(config['folder_out'], config['stage'])):
        os.makedirs(os.path.join(config['folder_out'], config['stage']))
    if config['train']:
        with open(os.path.join(config['folder_out'], config['stage'], 'config.yaml'), 'w') as f:
            yaml.safe_dump(config, f)
    set_seed(config['seed'])
    data_loaders, image_shape = get_data_loader(config)
    config['image_shape'] = image_shape
    net = get_model(config)
    pdb.set_trace()
    print('config:', config)
    if config['train']:
        train_model(config, data_loaders, net)
    test_model_latent(config, data_loaders, net)
    # test_model_predict(config, data_loaders, net)
    # test_model_statistics(config,data_loaders, net)
    # test_model_observe(config, data_loaders, net)


if __name__ == '__main__':
    main()
