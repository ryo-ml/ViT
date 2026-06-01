from pathlib import Path

from src.data.transform import get_transforms
from src.data.datasets import load_dataset, get_dataloader

BASE_DIR = Path(__file__).resolve().parent.parent
DS_DIR = BASE_DIR / 'data'

def test_dataloader() -> None:
    transforms = get_transforms(train=True)
    dataset = load_dataset(
        root=DS_DIR,
        train=True,
        transform=transforms,
    )
    dataloader = get_dataloader(
        dataset=dataset,
        batch_size=32,
        shuffle=True,
    )

    imgs, labels = next(iter(dataloader))

    # image size
    assert imgs.size(0) == 32
    assert imgs.size(1) == 3
    assert imgs.size(2) == 224
    assert imgs.size(3) == 224

    # label size
    assert labels.size(0) == 32
