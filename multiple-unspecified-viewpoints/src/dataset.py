from typing import Dict, Union
from multiprocessing import shared_memory

import h5py
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class BaseDataset(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
        image: Union[np.ndarray, None],
        segment: Union[np.ndarray, None],
        overlap: Union[np.ndarray, None],
        phase: str,
        rank: int,
    ) -> None:
        super().__init__()
        self.num_views_all = cfg.dataset.num_views_all
        self.permute = (phase == 'train')
        if rank == 0:
            image_ht, image_wd, image_ch = cfg.dataset.image_shape
            if 'num_views_data' in cfg.dataset:
                num_views_data = cfg.dataset.num_views_data
            else:
                num_views_data = cfg.dataset.num_views_all
            assert image.shape[0] == \
                segment.shape[0] == \
                overlap.shape[0]
            assert image.shape[1] == \
                segment.shape[1] == \
                overlap.shape[1] >= \
                num_views_data >= \
                cfg.dataset.num_views_all >= \
                cfg.dataset.num_views_max >= \
                cfg.dataset.num_views_min >= \
                1
            assert image.shape[2:] == (image_ht, image_wd, image_ch)
            assert segment.shape[2:] == (image_ht, image_wd)
            assert overlap.shape[2:] == (image_ht, image_wd)
            if not self.permute:
                image = image[:, :num_views_data]
                segment = segment[:, :num_views_data]
                overlap = overlap[:, :num_views_data]
            if cfg.dataset.num_views_all == 1:
                image = image.reshape(-1, 1, *image.shape[2:])
                segment = segment.reshape(-1, 1, *segment.shape[2:])
                overlap = overlap.reshape(-1, 1, *overlap.shape[2:])
            shm_image = shared_memory.SharedMemory(create=True,
                                                   size=image.nbytes)
            shm_segment = shared_memory.SharedMemory(create=True,
                                                     size=segment.nbytes)
            shm_overlap = shared_memory.SharedMemory(create=True,
                                                     size=overlap.nbytes)
            info_list = [
                shm_image.name,
                shm_segment.name,
                shm_overlap.name,
                image.shape,
                image.dtype,
                segment.shape,
                segment.dtype,
                overlap.shape,
                overlap.dtype,
            ]
        else:
            info_list = [None] * 9
        torch.distributed.broadcast_object_list(info_list)
        self.shm_image = shared_memory.SharedMemory(name=info_list[0])
        self.shm_segment = shared_memory.SharedMemory(name=info_list[1])
        self.shm_overlap = shared_memory.SharedMemory(name=info_list[2])
        self.image = np.ndarray(shape=info_list[3],
                                dtype=info_list[4],
                                buffer=self.shm_image.buf)
        self.segment = np.ndarray(shape=info_list[5],
                                  dtype=info_list[6],
                                  buffer=self.shm_segment.buf)
        self.overlap = np.ndarray(shape=info_list[7],
                                  dtype=info_list[8],
                                  buffer=self.shm_overlap.buf)
        if rank == 0:
            self.image[()] = image
            self.segment[()] = segment
            self.overlap[()] = overlap
        torch.distributed.barrier()
        self.image = torch.from_numpy(self.image)
        self.segment = torch.from_numpy(self.segment)
        self.overlap = torch.from_numpy(self.overlap)

    def __len__(self) -> int:
        return self.image.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        batch = {
            'image': self.image[idx],  # [V, H, W, C], uint8
            'segment': self.segment[idx],  # [V, H, W], uint8
            'overlap': self.overlap[idx],  # [V, H, W], uint8
        }
        if self.permute:
            indices = torch.randperm(batch['image'].shape[0])
            batch = {
                key: val[indices[:self.num_views_all]]
                for key, val in batch.items()
            }
        else:
            batch = {
                key: val[:self.num_views_all]
                for key, val in batch.items()
            }
        return batch


class DummyDataset(BaseDataset):
    def __init__(self, cfg: DictConfig, phase: str) -> None:
        rank = torch.distributed.get_rank()
        image_ht, image_wd, image_ch = cfg.dataset.image_shape
        num_views = cfg.dataset.num_views_data
        num_data = cfg.dataset.batch_size[phase] * 10
        if rank == 0:
            image = np.random.randint(
                low=0,
                high=8,
                size=[num_data, num_views, image_ht, image_wd, image_ch],
                dtype=np.uint8,
            )
            segment = np.random.randint(
                low=0,
                high=cfg.dataset.num_slots[phase],
                size=[num_data, num_views, image_ht, image_wd],
                dtype=np.uint8,
            )
            overlap = np.random.randint(
                low=0,
                high=cfg.dataset.num_slots[phase],
                size=[num_data, num_views, image_ht, image_wd],
                dtype=np.uint8,
            )
        else:
            image = None
            segment = None
            overlap = None
        super().__init__(cfg, image, segment, overlap, phase, rank)


class CustomDataset(BaseDataset):
    def __init__(self, cfg: DictConfig, phase: str) -> None:
        rank = torch.distributed.get_rank()
        alternative_dict = {
            'train': ['training'],
            'val': ['validation', 'valid'],
            'test': ['testing', 'test_1'],
            'general': ['generalization', 'test_2'],
        }
        path_dataset = hydra.utils.to_absolute_path(cfg.dataset.path)
        with h5py.File(path_dataset, 'r') as f:
            if phase in f:
                f_key = phase
            else:
                for f_key in alternative_dict[phase]:
                    if f_key in f:
                        break
                else:
                    raise KeyError
            if rank == 0:
                image = f[f_key]['image'][()]
                segment = f[f_key]['segment'][()]
                if 'overlap' in f[f_key]:
                    overlap = f[f_key]['overlap'][()]
                else:
                    assert cfg.dataset.seg_overlap == True
                    overlap = np.zeros_like(segment)
                if image.ndim == 4 and segment.ndim == 3 and overlap.ndim == 3:
                    image = image[:, None]
                    segment = segment[:, None]
                    overlap = overlap[:, None]
            else:
                image = None
                segment = None
                overlap = None
        super().__init__(cfg, image, segment, overlap, phase, rank)


class DataPrefetcher():
    def __init__(
        self,
        cfg: DictConfig,
        phase: str,
        dataloader: DataLoader,
    ) -> None:
        self.num_views_min = cfg.dataset.num_views_min
        self.num_views_max = cfg.dataset.num_views_max
        self.data_slots = cfg.dataset.num_slots[phase]
        self.sample_num_views = (phase == 'train')
        self.dataloader = dataloader
        self.data_iter = None
        self.idx_epoch = 0
        self.idx_batch = 0
        self.stream = torch.cuda.Stream()
        self.next_data = None
        self.setup()

    def setup(self, idx_epoch: int = 0, idx_batch: int = 0) -> None:
        self.dataloader.sampler.set_epoch(idx_epoch)
        self.data_iter = iter(self.dataloader)
        for _ in range(idx_batch):
            _ = next(self.data_iter)
        self.preload()
        return

    def preload(self) -> None:
        try:
            next_data = next(self.data_iter)
        except StopIteration:
            next_data = None
        else:
            with torch.cuda.stream(self.stream):
                if self.sample_num_views:
                    num_views_sel = torch.randint(
                        self.num_views_min,
                        self.num_views_max + 1,
                        size=[],
                    ).item()
                    next_data = {
                        key: val[:, :num_views_sel]
                        for key, val in next_data.items()
                    }
                next_data = {
                    key: val.cuda(non_blocking=True)
                    for key, val in next_data.items()
                }
                image = next_data['image']  # [B, V, H, W, C], uint8
                segment = next_data['segment']  # [B, V, H, W], uint8
                overlap = next_data['overlap']  # [B, V, H, W], uint8
                image = (image.to(torch.float32) / 255) * 2 - 1
                segment = segment[:, :, None].to(torch.int64)
                segment = torch.zeros(
                    [*segment.shape[:2], self.data_slots, *segment.shape[3:]],
                    dtype=torch.float32,
                    device=segment.device,
                ).scatter_(dim=2, index=segment, value=1.0)
                overlap = torch.gt(overlap[:, :, None], 1).to(torch.float32)
                next_data = {
                    'image': image,  # [B, V, H, W, C]
                    'segment': segment,  # [B, V, K, H, W]
                    'overlap': overlap,  # [B, V, 1, H, W]
                }
        self.next_data = next_data
        return

    def next(self) -> Dict[str, torch.Tensor]:
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        if data is not None:
            for val in data.values():
                val.record_stream(torch.cuda.current_stream())
        self.preload()
        return data


def get_data_prefetcher(cfg: DictConfig, phase: str) -> DataPrefetcher:
    if cfg.debug:
        dataset = DummyDataset(cfg, phase)
    else:
        dataset = CustomDataset(cfg, phase)
    world_size = torch.distributed.get_world_size()
    batch_size = cfg.dataset.batch_size[phase]
    assert batch_size % world_size == 0
    batch_size_local = batch_size // world_size
    is_train = (phase == 'train')
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size_local,
        num_workers=1,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        sampler=DistributedSampler(
            dataset,
            shuffle=is_train,
            drop_last=is_train,
        ),
    )
    data_prefetcher = DataPrefetcher(cfg, phase, dataloader)
    return data_prefetcher
