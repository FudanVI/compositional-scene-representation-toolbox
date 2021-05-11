import h5py
import numpy as np
import torch
import torch.utils.data as utils_data


class Dataset(utils_data.Dataset):

    def __init__(self, data):
        super(Dataset, self).__init__()
        self.images = torch.tensor(data['image'])
        self.segments = torch.tensor(data['segment'])
        self.overlaps = torch.tensor(data['overlap'])

    def __getitem__(self, idx):
        image = self.images[idx]
        segment = self.segments[idx]
        overlap = self.overlaps[idx]
        data = {'image': image, 'segment': segment, 'overlap': overlap}
        return data

    def __len__(self):
        return self.images.shape[0]


def get_data_loaders(config):
    image_shape = None
    data_loaders = {}
    with h5py.File(config['path_data'], 'r', libver='latest', swmr=True) as f:
        phase_list = [*f.keys()]
        if not config['train']:
            phase_list = [n for n in phase_list if n not in ['train', 'valid']]
        for phase in phase_list:
            data = {key: f[phase][key][()] for key in ['image', 'segment', 'overlap']}
            data['image'] = np.moveaxis(data['image'], -1, -3)
            if image_shape is None:
                image_shape = data['image'].shape[-3:]
            else:
                assert image_shape == data['image'].shape[-3:]
            data_loaders[phase] = utils_data.DataLoader(
                Dataset(data),
                batch_size=config['batch_size'],
                num_workers=0,
                shuffle=(phase == 'train'),
                drop_last=(phase == 'train'),
                pin_memory=True,
            )
    return data_loaders, image_shape
