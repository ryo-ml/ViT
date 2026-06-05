import hydra

from src.models.builder import build_model
from src.training.optimizer import get_optimizer
from src.training.loss import get_loss_fn
from src.training.trainer import Trainer

def test_trainer() -> None:
    with hydra.initialize(version_base=None, config_path='../configs'):
        cfg = hydra.compose(config_name='config')

    device = cfg.device
    model = build_model(cfg.model).to(device)
    optimizer = get_optimizer(cfg.optimizer, model.parameters())
    loss_fn = get_loss_fn(cfg.loss)

    trainer = Trainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
    )

    assert isinstance(trainer, Trainer)
