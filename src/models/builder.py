import hydra
from omegaconf import DictConfig
import torch

def build_model(model_cfg: DictConfig) -> torch.nn.Module:
    return hydra.utils.instantiate(model_cfg)
