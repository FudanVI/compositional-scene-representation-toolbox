import argparse
import os

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--num_runs', type=int)
    args = parser.parse_args()
    path_dst = os.path.join(args.folder, args.name)
    if os.path.islink(path_dst):
        os.remove(path_dst)
    best_loss = None
    path_src = None
    for idx_run in range(args.num_runs):
        name_run = f'{args.name}_run_{idx_run}'
        path_ckpt = os.path.join(args.folder, name_run, 'checkpoint',
                                 'train_curr.pth')
        load_dict = torch.load(path_ckpt)
        if best_loss is None or load_dict['best_loss'] < best_loss:
            best_loss = load_dict['best_loss']
            path_src = name_run
    os.symlink(path_src, path_dst)
    return


if __name__ == '__main__':
    main()
