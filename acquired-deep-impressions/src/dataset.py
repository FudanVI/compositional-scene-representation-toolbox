import h5py
import torch
import torch.utils.data as utils_data


class Dataset(utils_data.Dataset):

    def __init__(self, config, data):
        super(Dataset, self).__init__()
        self.images = torch.tensor(data['image'])
        self.segments = torch.tensor(data['segment'])
        self.overlaps = torch.tensor(data['overlap'])
        self.layers = torch.tensor(data['layers'])
        self.data_slots = self.segments.max() + 1
        self.seg_bck = config['seg_bck']
        self.seg_overlap = config['seg_overlap']

    def __getitem__(self, idx):
        image = self.images[idx].float() / 255
        segment = self.segments[idx].long().unsqueeze_(0)
        label = torch.zeros([self.data_slots, *segment.shape[1:]]).scatter_(0, segment, 1)
        if not self.seg_bck:
            label = label[1:]
        if not self.seg_overlap:
            valid = torch.le(self.overlaps[idx], 1).float().unsqueeze_(0)
            label *= valid
        label.unsqueeze_(-3)
        layers = self.layers[idx].float() / 255
        data = {'image': image, 'label': label, 'layers': layers}
        return data

    def __len__(self):
        return self.images.shape[0]


def get_data_loaders(config):
    with h5py.File(config['path_data'], 'r', libver='latest', swmr=True) as f:
        if config['mode_single']:
            phase_data = 'single'
            data_single = {key: f[phase_data][key][()] for key in f[phase_data]}
            data_all = {phase: data_single for phase in ['train', 'test']}
        else:
            phase_list = ['test', 'general']
            if config['train']:
                phase_list += ['train', 'valid']
            data_all = {}
            for phase in phase_list:
                if phase in ['train', 'valid']:
                    phase_data = '{}_{}'.format(phase, config['split_train'])
                elif phase in ['test', 'general']:
                    phase_data = '{}_{}'.format(phase, config['split_test'])
                else:
                    raise AssertionError
                data_all[phase] = {key: f[phase_data][key][()] for key in f[phase_data]}
    data_loaders, image_shape = {}, None
    for phase, data in data_all.items():
        if image_shape is None:
            image_shape = data['image'].shape[-3:]
        else:
            assert image_shape == data['image'].shape[-3:]
        data_loaders[phase] = utils_data.DataLoader(
            Dataset(config, data),
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            shuffle=(phase == 'train'),
            drop_last=(phase == 'train'),
            pin_memory=True,
        )
    return data_loaders, image_shape
