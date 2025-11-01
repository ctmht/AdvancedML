import os
from copy import deepcopy
from datetime import datetime
from itertools import batched, product
from typing import Any, Generator

import torch
import torch.nn as nn
import torch.optim as optim
from torch import device
from tqdm import tqdm, trange

from data_management import (
    reshape_dict_list,
    scale_dict_values,
    swap_nested_dict_axes,
)
from datasets import Dataset, get_MNIST, get_pixel_shift
from metrics import (
    MIG,
    MSE,
    BinaryCrossExtropy,
    ELBOLoss,
    GaussKLdiv,
    LatentMean,
    LatentStddev,
    Metrics,
    VAEMetric,
)
from models import BaseVAE, FeedForwardVAE
from plotting import test_performance_line, vae_visual_appraisal, variable_value_lines

GLOBAL_DEVICE = device("cuda") if torch.cuda.is_available() else device("cpu")
if not os.path.exists("data/images"):
    os.makedirs("data/images", exist_ok=True)
# Set the environment variable to disable NVML initialization errors
os.environ["NVIDIA_DISABLE_NVML"] = "1"


def vary_config_variable(
    config: dict, adjustments: dict[str, list] | list[dict[str, Any]]
):
    if isinstance(adjustments, dict):
        num_adj = len(list(adjustments.values())[0])
        adjustments = [
            {k: v[i] for k, v in adjustments.items()} for i in range(num_adj)
        ]

    for adj in adjustments:
        config_copy = deepcopy(config)
        for k, v in adj.items():
            config_copy[k] = v

        yield config_copy


class Schedule:  # I plan on adding stuff later, hence this class
    def __init__(self, number_of_epochs: int, optimizer: optim.Optimizer) -> None:
        self.n_epochs = number_of_epochs
        self.schedule: dict[int, dict[str, Any]] = {}
        self.optimizer = optimizer
        self.epoch: int = 0

    def manual_schedule(
        self, parameter_schedule: dict[str, dict] | dict[int, str]
    ) -> None:
        """
        Add an optimizer hyperparameter schedule manually
        """
        self.schedule = parameter_schedule
        if isinstance(list(parameter_schedule.keys())[0], str):
            self.schedule = swap_nested_dict_axes(parameter_schedule)

    def adjust_optimizer(self, epoch: int) -> None:
        if epoch not in self.schedule:
            return

        for g in self.optimizer.param_groups:
            for parameter_name, value in self.schedule[epoch].items():
                g[parameter_name] = value

    def __iter__(self):
        return self._generator()

    def _generator(self) -> Generator[int, None, None]:
        for i in trange(self.n_epochs, desc="Running experiment", leave=True):
            self.epoch = i
            self.adjust_optimizer(i)
            yield i


def run_test(dataset, model, loss_func, metrics):
    model.eval()
    test_losses = []
    test_loader = dataset.test_loader
    label_converter = dataset.label_converter
    metrics.reset_metrics()

    for i, label in tqdm(test_loader, desc="running test", leave=False):
        i = i.to(GLOBAL_DEVICE)
        label = label_converter(label)
        out, latent = model(i)
        loss = loss_func(i, latent, out, label)
        test_losses.append(loss.detach().item())
        metrics(i, latent, out, label)

    output = metrics.mean_metrics()
    output["loss"] = (sum(test_losses) / len(test_losses),)
    metrics.recorded_values["loss"] = metrics.recorded_values.get("loss", []) + [
        output["loss"]
    ]
    return output


def run_train(dataset, model, schedule, loss_func, metrics: Metrics):
    model.train()
    train_losses = []
    loss_sum = 0
    count = 0
    train_loader = dataset.train_loader
    label_converter = dataset.label_converter
    optimizer = schedule.optimizer
    metrics.archive_metrics()

    train_bar = tqdm(train_loader, desc=f"epoch {schedule.epoch}", leave=False)
    for i, label in train_bar:
        optimizer.zero_grad()
        i = i.to(GLOBAL_DEVICE)
        label = label_converter(label)
        out, latent = model(i)
        loss = loss_func(i, latent, out, label)
        metrics(i, latent, out, label, prefix="train ", exclude=["MIG"])
        train_losses.append(loss.item())
        loss.backward()
        loss_sum += loss.item()
        count += 1
        optimizer.step()
        train_bar.set_postfix({"loss": f"{loss_sum / count:.3e}"})

    output = metrics.mean_metrics()
    output["loss"] = (sum(train_losses) / len(train_losses),)
    metrics.recorded_values["loss"] = metrics.recorded_values.get("loss", []) + [
        output["loss"]
    ]
    return output


def run_epoch(
    dataset: Dataset,
    model: nn.Module,
    schedule: Schedule,
    loss_func: VAEMetric,
    metrics: Metrics,
):
    run_train(dataset, model, schedule, loss_func, metrics)
    return run_test(dataset, model, loss_func, metrics)


def run_experiment(
    dataset: Dataset,
    schedule: Schedule,
    model: BaseVAE,
    loss_func: VAEMetric,
    metrics: Metrics,
):
    metric_values = [run_test(dataset, model, loss_func, metrics)]

    for _ in schedule:
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


def create_log_directory(test_name: str) -> None:
    log_dir = f"data/logs/automatic/{test_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir + "/metrics", exist_ok=True)
    os.makedirs(log_dir + "/images", exist_ok=True)


def experiment_from_config(config: dict, verbose: bool = False) -> None:
    """
    Runs a test with an MLP-based VAE configurable with a dictionary
    config keys:
        experiment_name: str
        image_shape: tuple[int, int]
        batch_size: int
        latent_space_size: int
        width: int
        depth: int
        mnist: bool # alternative is pixel
        optimizer: type[optim.Optimizer]
        n_epochs: int
        schedule: dict[str, dict[int, float]]
        loss_func: VAEMetric
    """
    # We only use two datasets, so we can build them internally
    image_shape = config["image_shape"]
    n_pixels = image_shape[0] * image_shape[1]
    if config["mnist"]:
        dataset = get_MNIST(config["batch_size"], 256 * 4)
    else:
        dataset = get_pixel_shift(
            image_shape, (16384, 2048), config["batch_size"], 2048
        )

    # Likewise we only use an MLP
    model = FeedForwardVAE(
        n_pixels,
        config["latent_space_size"],
        config["width"],
        config["depth"],
        in_channels=1,
    ).to(GLOBAL_DEVICE)

    # The optimizer setup is largely determined in the `Schedule` object
    optimizer = config["optimizer"](model.parameters())
    schedule = Schedule(number_of_epochs=config["n_epochs"], optimizer=optimizer)
    schedule.manual_schedule(config["schedule"])
    loss_func = config["loss_func"]

    metrics = Metrics(
        {
            "kl_div": GaussKLdiv(),
            "mse": MSE(),
            "bce": BinaryCrossExtropy(),
            "MIG": MIG(),
            "latent_mean": LatentMean(),
            "latent_stddev": LatentStddev(),
        },
        ["train ", ""],
    )
    name = config["experiment_name"]
    create_log_directory(name)
    metric_values = run_experiment(dataset, schedule, model, loss_func, metrics)

    metrics.archive_metrics()
    metrics.dump(
        f"data/logs/automatic/{name}/metrics/{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        str(config),
    )

    # visualisation
    test_performance_line(
        {
            "elbo": metric_values["loss"],
            "mse": metric_values["mse"],
            "bce": metric_values["bce"],
        },
        path=f"{name}/images/losses.pdf",
        show=verbose,
    )
    test_performance_line(
        {"kl_div": metric_values["kl_div"]},
        path=f"{name}/images/kl_div.pdf",
        show=verbose,
    )
    test_performance_line(
        {
            "latent mean": metric_values["latent_mean"],
            "latent stddev": metric_values["latent_stddev"],
        },
        path=f"{name}/images/latent_stats.pdf",
        show=verbose,
    )
    test_performance_line(
        {
            "latent mean": metric_values["latent_mean"],
        },
        log=False,
        path=f"{name}/images/latent_mean.pdf",
        show=verbose,
    )
    test_performance_line(
        {
            "MIG": metric_values["MIG"],
        },
        log=False,
        path=f"{name}/images/mig.pdf",
        show=verbose,
    )

    example_images = [dataset.test_dataset[i][0].to(GLOBAL_DEVICE) for i in range(10)]
    vae_visual_appraisal(
        model,
        name,
        example_images,
        GLOBAL_DEVICE,
        show=verbose,
    )


def main():
    config = {
        "experiment_name": "pixel_grid_test",
        "image_shape": (28, 28),
        "batch_size": 256 + 128,
        "mnist": True,
        "width": 512,
        "depth": 4,
        "latent_space_size": 20,
        "optimizer": optim.Adam,
        "schedule": {"lr": scale_dict_values({0: 1e-4, 100: 5e-4, 200: 1e-5}, 5e0)},
        "n_epochs": 3,
        "loss_func": ELBOLoss(0.0),
    }

    grid_setup = list(product([0, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 2], [4, 10, 100]))
    config_adjustments = {
        "experiment_name": [f"MNIST_mig_test3/b={b}zs={zs}" for b, zs in grid_setup],
        "loss_func": [ELBOLoss(b) for b, _ in grid_setup],
        "latent_space_size": [zs for _, zs in grid_setup],
    }

    configs = list(vary_config_variable(config, config_adjustments))
    for config in tqdm(configs, desc="Running multi-config experiment"):
        experiment_from_config(config, False)


if "__main__" in __name__:
    main()
