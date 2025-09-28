import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor, device
from typing import Callable
from tqdm import trange, tqdm

from models import ExampleVAE, BaseVAE
from metrics import ELBOLoss, VAEMetric
from plotting import vae_visual_appraisal, test_performance_line
from datasets import Dataset, get_MNIST


GLOBAL_DEVICE = device("cuda") if torch.cuda.is_available() else device("cpu")


class Schedule:  # I plan on adding stuff later, hence this class
    def __init__(self, number_of_epochs: int, optimizer: optim.Optimizer) -> None:
        self.n_epochs = number_of_epochs
        self.lr = {}
        self.optimizer = optimizer

    def adjust_optimizer(self, lr: float) -> None:
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def __iter__(self):
        return self._generator()

    def _generator(self) -> int:
        for i in trange(self.n_epochs):
            if i in self.lr:
                self.adjust_optimizer(self.lr[i])
            yield i


def run_test(test_loader, model, loss_func):
    model.eval()
    test_losses = []
    for i, label in test_loader:
        i = i.to(GLOBAL_DEVICE)
        out, latent = model(i)
        loss = loss_func(i, latent, out)
        test_losses.append(loss.detach().item())

    test_loss = sum(test_losses) / len(test_losses)
    return test_loss


def run_train(train_loader, model, optimizer, loss_func):
    model.train()
    train_losses = []
    train_bar = tqdm(train_loader)
    for i, label in train_bar:
        optimizer.zero_grad()
        i = i.to(GLOBAL_DEVICE)
        out, latent = model(i)
        loss = loss_func(i, latent, out)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        train_bar.set_postfix({"loss": f"{loss.item():.3f}"})

    return train_losses


def run_epoch(
    train_loader,
    test_loader,
    model: nn.Module,
    optimizer,
    loss_func: Callable[[Tensor, Tensor, Tensor], Tensor],
):
    run_train(train_loader, model, optimizer, loss_func)
    return run_test(test_loader, model, loss_func)


def run_experiment(
    dataset: Dataset,
    schedule: Schedule,
    model: BaseVAE,
    loss_func: VAEMetric,
):
    test_losses = [run_test(dataset.test_loader, model, loss_func)]
    for epoch in schedule:
        test_losses.append(
            run_epoch(
                dataset.train_loader,
                dataset.test_loader,
                model,
                schedule.optimizer,
                loss_func,
            )
        )
    return test_losses


if "__main__" in __name__:
    dataset = get_MNIST(8, 256)
    model = ExampleVAE(
        784, latent_size=140, in_channels=1, inter_channels=[64, 128]
    ).to(GLOBAL_DEVICE)
    schedule = Schedule(
        number_of_epochs=12, optimizer=optim.AdamW(model.parameters(), lr=1e-5)
    )
    loss_func = ELBOLoss(0)
    test_losses = run_experiment(dataset, schedule, model, loss_func)

    print(test_losses)
    # visualisation
    if not os.path.exists("data/images"):
        os.makedirs("data/images")

    test_performance_line(test_losses)
    example_images = [dataset.test_dataset[i][0].to(GLOBAL_DEVICE) for i in range(10)]
    vae_visual_appraisal(model, "MNIST", example_images, GLOBAL_DEVICE)
