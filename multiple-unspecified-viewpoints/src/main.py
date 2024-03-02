import os

import hydra
import torch
from dataset import get_data_prefetcher
from model_def import Model
from model_run import run_testing, run_training
from omegaconf import DictConfig


@hydra.main(
    config_path='config',
    config_name='dummy',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    assert cfg.precision in ['fp32', 'tf32', 'bf16']
    if cfg.precision == 'fp32':
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.distributed.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        module=Model(cfg).to(local_rank),
        device_ids=[local_rank],
    )
    if cfg.training:
        data_prefetchers = {
            phase: get_data_prefetcher(cfg, phase)
            for phase in ['train', 'val']
        }
        run_training(cfg, model, data_prefetchers)
    if cfg.testing:
        data_prefetchers = {
            phase: get_data_prefetcher(cfg, phase)
            for phase in ['test', 'general']
        }
        run_testing(cfg, model, data_prefetchers)
    torch.distributed.destroy_process_group()
    return


if __name__ == '__main__':
    main()
