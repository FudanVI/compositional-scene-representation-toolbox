import argparse
import datetime
import os
import tensorflow as tf
import yaml
from dataset import get_data_loaders
from model import get_model
from run_model import train_model, test_model
from utils import unpack_row_col


def get_config():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    _ = [tf.config.experimental.set_memory_growth(val, True) for val in gpus]
    if len(gpus) == 0:
        raise ValueError
    elif len(gpus) == 1:
        strategy = tf.distribute.OneDeviceStrategy('gpu:0')
    else:
        strategy = tf.distribute.MirroredStrategy()
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config')
    parser.add_argument('--path_pretrain')
    parser.add_argument('--folder_data')
    parser.add_argument('--folder_log')
    parser.add_argument('--folder_out')
    parser.add_argument('--timestamp')
    parser.add_argument('--num_tests', type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--use_timestamp', action='store_true')
    parser.add_argument('--folder_ckpt', default='ckpt')
    parser.add_argument('--folder_model', default='model')
    parser.add_argument('--name_save', default='save')
    args = parser.parse_args()
    with open(args.path_config) as f:
        config = yaml.safe_load(f)
    for key, val in args.__dict__.items():
        if key not in config or val is not None:
            config[key] = val
    feat_ht, feat_wd = config['image_shape'][:-1]
    for strides in config['dec_strides']:
        row_stride, col_stride = unpack_row_col(strides)
        assert feat_ht % row_stride == 0
        assert feat_wd % col_stride == 0
        feat_ht //= row_stride
        feat_wd //= col_stride
    config['feat_shape'] = [feat_ht, feat_wd]
    if config['debug']:
        config['ckpt_intvl'] = 1
        strategy = tf.distribute.OneDeviceStrategy('gpu:0')
    if config['resume']:
        config['train'] = True
    if config['timestamp'] is None:
        config['timestamp'] = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    if config['use_timestamp']:
        for key in ['folder_log', 'folder_out']:
            config[key] = os.path.join(config[key], config['timestamp'])
    if config['train'] and not config['resume']:
        for key in ['folder_log', 'folder_out']:
            if os.path.exists(config[key]):
                raise FileExistsError(config[key])
            os.makedirs(config[key])
        with open(os.path.join(config['folder_out'], 'config.yaml'), 'w') as f:
            yaml.safe_dump(config, f)
    return config, strategy


def main():
    config, strategy = get_config()
    data_loaders = get_data_loaders(strategy, config)
    net, optimizer, checkpoint = get_model(strategy, config)
    if config['train']:
        train_model(strategy, config, data_loaders, net, optimizer, checkpoint)
    test_model(strategy, config, data_loaders, net)
    return


if __name__ == '__main__':
    main()
