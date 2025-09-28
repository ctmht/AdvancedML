import torch
from torchvision import datasets, transforms
from collections import namedtuple


NUMBER_OF_THREADS = 16
DATASET_DIR = "data/datasets"
Dataset = namedtuple(
    "Dataset", ["train_dataset", "test_dataset", "train_loader", "test_loader"]
)


def get_MNIST(train_batch_size: int, test_batch_size: int) -> Dataset:
    train_dataset = datasets.MNIST(
        root=DATASET_DIR, train=True, download=True, transform=transforms.ToTensor()
    )
    test_dataset = datasets.MNIST(
        root=DATASET_DIR, train=False, download=True, transform=transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=NUMBER_OF_THREADS,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        num_workers=NUMBER_OF_THREADS,
    )
    return Dataset(train_dataset, test_dataset, train_loader, test_loader)
