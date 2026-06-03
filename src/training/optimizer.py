from typing import Iterator
from omegaconf import DictConfig
import hydra
import torch

def get_optimizer(
    optimizer_cfg: DictConfig,
    params: Iterator[torch.nn.Parameter],
) -> torch.optim.Optimizer:
    return hydra.utils.instantiate(optimizer_cfg, params=params)
