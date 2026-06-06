from pathlib import Path
from torchvision.transforms.v2 import Compose
from torchvision.datasets import CIFAR100
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

from src.data.transform import get_transforms

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
        num_workers=2,
        pin_memory=True,
    )
    return dataloader

def get_dataloaders(
    root: Path,
    train_val_split: list[float],
    seed: int,
    batch_size: int,
) -> tuple[DataLoader]:
    assert len(train_val_split) == 2 and sum(train_val_split) == 1.0, 'train_val_split must contain two values that sum to 1.'

    train_val_set = load_dataset(
        root,
        train=True,
        transform=get_transforms(train=True)
    )

    train_indices, val_indices = train_test_split(
        range(len(train_val_set)),
        test_size=train_val_split[1],
        train_size=train_val_split[0],
        random_state=seed,
        shuffle=True,
        stratify=train_val_set.targets,
    )

    train_set = Subset(train_val_set, train_indices)
    val_set = Subset(train_val_set, val_indices)
    test_set = load_dataset(
        root,
        train=False,
        transform=get_transforms(train=False)
    )

    train_loader = get_dataloader(train_set, batch_size, shuffle=True)
    val_loader = get_dataloader(val_set, batch_size, shuffle=False)
    test_loader = get_dataloader(test_set, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
