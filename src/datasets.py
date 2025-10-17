import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision import datasets, transforms
from collections import namedtuple
from random import randrange


NUMBER_OF_THREADS = 8
DATASET_DIR = "data/datasets"
Dataset = namedtuple(
    "Dataset",
    ["train_dataset", "test_dataset", "train_loader", "test_loader", "label_converter"],
)


def simple_converter(x):
    return x


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
    return Dataset(
        train_dataset,
        test_dataset,
        train_loader,
        test_loader,
        lambda x: F.one_hot(x.clone().long(), 10).float(),
    )


def create_pixel_shift_image(shape: tuple, index: int | tuple) -> tuple[Tensor, Tensor]:
    image = torch.zeros(size=shape)
    if isinstance(index, int):
        index = (index // shape[1], index % shape[0])
    image[index[0], index[1]] = 1.0
    return image.unsqueeze(0), torch.tensor(index)


def get_pixel_shift(
    shape: tuple[int, int],
    num_examples: tuple[int, int],
    train_batch_size: int,
    test_batch_size: int,
) -> Dataset:
    max_index = shape[0] * shape[1] - 1
    train_dataset = [
        create_pixel_shift_image(shape, randrange(0, max_index))
        for _ in range(num_examples[0])
    ]
    test_dataset = [
        create_pixel_shift_image(shape, randrange(0, max_index))
        for _ in range(num_examples[1])
    ]

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
    return Dataset(
        train_dataset, test_dataset, train_loader, test_loader, simple_converter
    )
