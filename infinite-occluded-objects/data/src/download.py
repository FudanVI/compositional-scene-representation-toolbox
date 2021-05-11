import argparse
import os
import torchvision


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_downloads', default='downloads')
    args = parser.parse_args()
    if not os.path.exists(args.folder_downloads):
        os.mkdir(args.folder_downloads)
    torchvision.datasets.MNIST(args.folder_downloads, download=True)
