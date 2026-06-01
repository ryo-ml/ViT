from pathlib import Path
from torchvision.transforms.v2 import Compose
from torchvision.datasets import CIFAR100
from torch.utils.data import Dataset, DataLoader

def load_dataset(
        root: Path,
        train: bool,
        transform: Compose,
    ) -> Dataset:
    dataset = CIFAR100(
        root=root,
        train=train,
        transform=transform,
        download=True,
    )
    return dataset

def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return dataloader
