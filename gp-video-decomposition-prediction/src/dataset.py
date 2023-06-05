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
        if self.images.ndim == 4 and self.segment.ndim == 3 and self.overlap.ndim == 3:
            self.images = self.images[:, None]
            self.segments = self.segment[:, None]
            self.overlaps = self.overlap[:, None]

        assert self.images.ndim == 5
        assert self.segments.ndim == 4
        assert self.overlaps.ndim == 4

    def __getitem__(self, idx):
        image = self.images[idx]
        segment = self.segments[idx]
        overlap = self.overlaps[idx]
        data = {'image': image, 'segment': segment, 'overlap': overlap}
        return data

    def __len__(self):
        return self.images.shape[0]


def get_data_loader(config):
    image_shape = None
    datasets = {}
    with h5py.File(config['path_data'], 'r', libver='latest', swmr=True) as f:
        phase_list = [*f.keys()]
        if not config['train']:
            phase_list = [val for val in phase_list if val not in ['train', 'valid']]
        index_sel = slice(config['batch_size']) if config['debug'] else ()
        for phase in phase_list:
            data = {key: f[phase][key][index_sel] for key in f[phase] if key not in ['layers', 'masks']}
            data['image'] = np.moveaxis(data['image'], -1, -3)
            if image_shape is None:
                image_shape = data['image'].shape[-3:]
            else:
                assert image_shape == data['image'].shape[-3:]
            datasets[phase] = Dataset(data)

    if 'train' in datasets and 'valid' not in datasets:
        data['valid'] = datasets['train']
    data_loaders = {}
    for key, val in datasets.items():
        data_loaders[key] = utils_data.DataLoader(
            val,
            batch_size=config['batch_size'] if phase in ['train', 'valid'] else config['test_batch_size'],
            num_workers=1,
            shuffle=(key == 'train'),
            drop_last=(key == 'train'),
            pin_memory=True,
        )

    return data_loaders, image_shape
