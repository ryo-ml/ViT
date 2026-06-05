import hydra
import torch

from src.models.builder import build_model
from src.training.optimizer import get_optimizer
from src.training.loss import get_loss_fn

def test_build_model() -> None:
    with hydra.initialize(version_base=None, config_path='../configs'):
        cfg = hydra.compose(config_name='config')
    model = build_model(cfg.model)
    assert isinstance(model, torch.nn.Module)

def test_get_optimizer() -> None:
    with hydra.initialize(version_base=None, config_path='../configs'):
        cfg = hydra.compose(config_name='config')
    model = build_model(cfg.model)
    optimizer = get_optimizer(cfg.optimizer, model.parameters())
    assert isinstance(optimizer, torch.optim.Optimizer)

def test_get_loss_fn() -> None:
    with hydra.initialize(version_base=None, config_path='../configs'):
        cfg = hydra.compose(config_name='config')
    loss_fn = get_loss_fn(cfg.loss)
    assert isinstance(loss_fn, torch.nn.Module)
