from omegaconf import DictConfig
import hydra
import torch

def get_loss_fn(loss_cfg: DictConfig) -> torch.nn.Module:
    return hydra.utils.instantiate(loss_cfg)
