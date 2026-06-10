from tqdm import tqdm
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader

from src.utils.logger import get_logger
from src.utils.metrics_logger import MetricsLogger
from src.utils.checkpoint import Checkpointer

logger = get_logger(__name__)

class Trainer:
    def __init__(
        self,
        cfg: DictConfig,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        metrics_logger: MetricsLogger,
        device: torch.device,
        optimizer: torch.optim.Optimizer | None = None,
        checkpointer: Checkpointer | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics_logger = metrics_logger
        self.checkpointer = checkpointer
        self.device = device

        self.step = 0

    def train_one_epoch(
        self,
        train_loader: DataLoader,
    ) -> None:
        assert self.optimizer is not None, 'optimizer is required for training'

        self.model.train()

        interval_loss = 0.0
        interval_samples = 0
        total_loss = 0.0
        total_samples = 0

        batch_pbar = tqdm(train_loader, desc='Training', position=1, leave=False)
        for batch in batch_pbar:
            imgs, labels = batch
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(imgs)
            loss = self.loss_fn(logits, labels)
            loss.backward()
            self.optimizer.step()

            # stats
            batch_size = labels.size(0)
            interval_loss += loss.item() * batch_size
            interval_samples += batch_size

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            self.step += 1

            # log per interval
            if self.step % self.cfg.training.log_interval == 0:
                self.metrics_logger.log(
                    self.step,
                    {
                        'train/loss': interval_loss / interval_samples,
                    }
                )

                interval_loss = 0.0
                interval_samples = 0

        # log per epoch
        epoch_loss = total_loss / total_samples
        self.metrics_logger.log(
            self.step,
            {
                'train/epoch_loss': epoch_loss,
            }
        )
        logger.info(f'[step {self.step}] train/epoch_loss={epoch_loss:.4f}')

    def valid(
        self,
        val_loader: DataLoader,
    ) -> None:
        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0

        batch_pbar = tqdm(val_loader, desc='Valid', position=1, leave=False)
        for batch in batch_pbar:
            imgs, labels = batch
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            with torch.no_grad():
                logits = self.model(imgs)
                loss = self.loss_fn(logits, labels)
                preds = torch.argmax(logits, dim=1)

                # stats
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_acc += (preds == labels).sum().item()
                total_samples += batch_size

        mean_loss = total_loss / total_samples
        mean_acc = total_acc / total_samples

        # log
        metrics = {
            'val/loss': mean_loss,
            'val/acc': mean_acc,
        }
        self.metrics_logger.log(self.step, metrics)
        logger.info(f'[step {self.step}] val/loss={mean_loss:.4f} val/acc={mean_acc:.4f}')

        # checkpoint
        if self.checkpointer is not None:
            self.checkpointer.save(self.model, self.step, metrics, filename='last.pth')
            if self.checkpointer.is_best(mean_acc):
                self.checkpointer.save(self.model, self.step, metrics, filename='best.pth')
                logger.info(f'[step {self.step}] New best model saved (val/acc={mean_acc:.4f})')

    def test(
        self,
        test_loader: DataLoader,
    ) -> None:
        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0

        batch_pbar = tqdm(test_loader, desc='Test', position=1)
        for batch in batch_pbar:
            imgs, labels = batch
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            with torch.no_grad():
                logits = self.model(imgs)
                loss = self.loss_fn(logits, labels)
                preds = torch.argmax(logits, dim=1)

                # stats
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_acc += (preds == labels).sum().item()
                total_samples += batch_size

        test_loss = total_loss / total_samples
        test_acc = total_acc / total_samples
        self.metrics_logger.log(
            self.step,
            {
                'test/loss': test_loss,
                'test/acc': test_acc,
            }
        )
        logger.info(f'[step {self.step}] test/loss={test_loss:.4f} test/acc={test_acc:.4f}')

    def run(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        # log baseline before training
        self.valid(val_loader)

        epoch_pbar = tqdm(range(self.cfg.training.num_epochs), desc='Epochs', position=0)
        for _ in epoch_pbar:
            self.train_one_epoch(train_loader)
            self.valid(val_loader)
