import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor, device
from typing import Any, Callable, Generator, Hashable
from tqdm import trange, tqdm

from models import ExampleVAE, BaseVAE, FeedForwardVAE
from metrics import (
    MSE,
    BinaryCrossExtropy,
    ELBOLoss,
    GaussKLdiv,
    LatentMean,
    LatentStddev,
    VAEMetric,
    Metrics,
    MIG,
)
from plotting import vae_visual_appraisal, test_performance_line
from datasets import Dataset, get_MNIST, get_pixel_shift


GLOBAL_DEVICE = device("cuda") if torch.cuda.is_available() else device("cpu")


def swap_nested_dict_axes(dictionary: dict) -> dict:
    keys = []
    for nested_dicts in dictionary.values():
        keys += list(nested_dicts.keys())

    keys = set(keys)
    output = {}
    for key in keys:
        output[key] = {
            outer_key: nested_dict[key]
            for outer_key, nested_dict in dictionary.items()
            if key in nested_dict
        }

    return output


def scale_dict_values(dictionary: dict, scaling: int | float) -> dict:
    return {k: v * scaling for k, v in dictionary.items()}


def reshape_dict_list(dict_list: list[dict]) -> dict[Hashable, list]:
    keys = list(dict_list[0].keys())
    output = {i: [] for i in keys}
    for i in dict_list:
        for k, v in i.items():
            output[k].append(v)
    return output


class Schedule:  # I plan on adding stuff later, hence this class
    def __init__(self, number_of_epochs: int, optimizer: optim.Optimizer) -> None:
        self.n_epochs = number_of_epochs
        self.schedule: dict[int, dict[str, Any]] = {}
        self.optimizer = optimizer
        self.epoch: int = 0

    def manual_schedule(
        self, parameter_schedule: dict[str, dict] | dict[int, str]
    ) -> None:
        if isinstance(list(parameter_schedule.keys())[0], str):
            self.schedule = swap_nested_dict_axes(parameter_schedule)
        else:
            self.schedule = parameter_schedule

    def adjust_optimizer(self, epoch: int) -> None:
        if epoch not in self.schedule:
            return

        for g in self.optimizer.param_groups:
            for parameter_name, value in self.schedule[epoch].items():
                g[parameter_name] = value

    def __iter__(self):
        return self._generator()

    def _generator(self) -> Generator[int, None, None]:
        for i in trange(self.n_epochs):
            self.epoch = i
            self.adjust_optimizer(i)
            yield i


def run_test(test_loader, model, loss_func, metrics):
    model.eval()
    test_losses = []
    test_loader = dataset.test_loader
    label_converter = dataset.label_converter
    metrics.reset_metrics()
    for i, label in tqdm(test_loader, desc="running test"):
        i = i.to(GLOBAL_DEVICE)
        label = label_converter(label)
        out, latent = model(i)
        loss = loss_func(i, latent, out, label)
        test_losses.append(loss.detach().item())
        metrics(i, latent, out, label)
    output = metrics.mean_metrics()
    output["loss"] = (sum(test_losses) / len(test_losses),)
    return output


def run_train(dataset, model, schedule, loss_func, metrics: Metrics):
    model.train()
    train_losses = []
    loss_sum = 0
    count = 0
    train_loader = dataset.train_loader
    label_converter = dataset.label_converter
    train_bar = tqdm(train_loader, desc=f"epoch {schedule.epoch}")
    optimizer = schedule.optimizer
    metrics.reset_metrics()
    for i, label in train_bar:
        optimizer.zero_grad()
        i = i.to(GLOBAL_DEVICE)
        label = label_converter(label)
        out, latent = model(i)
        loss = loss_func(i, latent, out, label)
        metrics(i, latent, out, label)
        train_losses.append(loss.item())
        loss.backward()
        loss_sum += loss.item()
        count += 1
        optimizer.step()
        train_bar.set_postfix({"loss": f"{loss_sum / count:.3e}"})
    output = metrics.mean_metrics()
    output["loss"] = (sum(train_losses) / len(train_losses),)
    return output


def run_epoch(
    dataset: Dataset,
    model: nn.Module,
    schedule: Schedule,
    loss_func: VAEMetric,
    metrics: Metrics,
):
    return run_train(dataset, model, schedule, loss_func, metrics)
    # return run_test(test_loader, model, loss_func, metrics)


def run_experiment(
    dataset: Dataset,
    schedule: Schedule,
    model: BaseVAE,
    loss_func: VAEMetric,
    metrics: Metrics,
):
    metric_values = [run_test(dataset, model, loss_func, metrics)]

    for epoch in schedule:
        metric_values.append(
            run_epoch(
                dataset,
                model,
                schedule,
                loss_func,
                metrics,
            )
        )
    return reshape_dict_list(metric_values)


if "__main__" in __name__:
    image_shape = (28, 28)
    n_pixels = image_shape[0] * image_shape[1]
    dataset = get_MNIST(256 * 2, 256)
    # dataset = get_pixel_shift(image_shape, (16384, 2048), 256 + 128, 2048)
    model = FeedForwardVAE(n_pixels, 20, 4096 * 2, 3, in_channels=1).to(GLOBAL_DEVICE)
    schedule = Schedule(
        number_of_epochs=250,
        optimizer=optim.AdamW(model.parameters(), lr=1e-3),
    )
    schedule.manual_schedule(
        {
            "lr": scale_dict_values(
                {
                    0: 3e-4,
                    # 55: 3e-5,
                    # 45: 3e-6,
                    # 50: 1e-6,
                    # 55: 3e-7,
                    # 200: 1e-5,
                    # 300: 1e-3,
                    # 1000: 1e-4,
                    # 2050: 1e-3,
                    # 300: 6e-4,
                    # 400: 3e-4,
                },
                1e0,
            ),
            # "momentum": {120: 0.01},
        }
    )
    loss_func = ELBOLoss(0.000000)
    metrics = Metrics(
        {
            "kl_div": GaussKLdiv(),
            "mse": MSE(),
            "bce": BinaryCrossExtropy(),
            # "MIG": MIG(),
            "latent_mean": LatentMean(),
            "latent_stddev": LatentStddev(),
        }
    )
    metric_values = run_experiment(dataset, schedule, model, loss_func, metrics)

    print(metric_values)
    # visualisation
    if not os.path.exists("data/images"):
        os.makedirs("data/images")

    test_performance_line(
        {
            "elbo": metric_values["loss"],
            "mse": metric_values["mse"],
            "bce": metric_values["bce"],
        }
    )
    test_performance_line({"kl_div": metric_values["kl_div"]})
    test_performance_line(
        {
            "latent mean": metric_values["latent_mean"],
            "latent stddev": metric_values["latent_stddev"],
        }
    )
    test_performance_line(
        {
            "latent mean": metric_values["latent_mean"],
        },
        log=False,
    )
    # test_performance_line({"mig": metric_values["MIG"]})
    example_images = [dataset.test_dataset[i][0].to(GLOBAL_DEVICE) for i in range(10)]
    vae_visual_appraisal(
        model, "MNIST_linear_full_beta-0", example_images, GLOBAL_DEVICE
    )
    # MNIST_linear: bs=32*32*8, width=256, depth=2, ls=10, lr=3 * 1e-4a, ELBO(0.2 * 1e-11) = 0.0275
    # MNIST_linear: bs=32*32*8, width=256, depth=2, ls=50, lr=3 * 1e-4a, ELBO(0.2 * 1e-11) = 0.0160
    # MNIST_linear: bs=32*32*8, width=1024, depth=2, ls=784, lr=3 * 1e-4a, ELBO(0.2 * 1e-11) = 0.0042
    # MNIST_linear: bs=32*32*8, width=1024, depth=2, ls=784, lr=3 * 1e-4a, ELBO(0) = 0.0030
    # MNIST_linear: bs=32*32*8, width=1024, depth=2, ls=784, lr=3 * 1e-4a, ELBO(1e-10) = 0.0147
    # MNIST_linear: bs=32*32*8, width=1024, depth=2, ls=784, lr=3 * 1e-4a, ELBO(1e-9) = 0.0287
