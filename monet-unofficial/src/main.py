import argparse
import datetime
import os
import tensorflow.compat.v1 as tf
import yaml
from dataset import get_data_loaders
from model import get_model
from run_model import train_model, test_model


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config')
    parser.add_argument('--folder_data')
    parser.add_argument('--folder_log')
    parser.add_argument('--folder_out')
    parser.add_argument('--timestamp')
    parser.add_argument('--num_tests', type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--use_timestamp', action='store_true')
    parser.add_argument('--ckpt_curr', default='checkpoint_curr')
    parser.add_argument('--ckpt_best', default='checkpoint_best')
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
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    config = get_config()
    if config['train'] and not config['resume']:
        for key in ['folder_log', 'folder_out']:
            if os.path.exists(config[key]):
                raise FileExistsError(config[key])
            os.makedirs(config[key])
        with open(os.path.join(config['folder_out'], 'config.yaml'), 'w') as f:
            yaml.safe_dump(config, f)
    strategy = tf.distribute.MirroredStrategy()
    data_loaders = get_data_loaders(strategy, config)
    net = get_model(config)
    if config['train']:
        train_model(strategy, config, data_loaders, net)
    test_model(strategy, config, data_loaders, net)
    return


if __name__ == '__main__':
    main()
