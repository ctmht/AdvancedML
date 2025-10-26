import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor, device
from typing import Any, Callable, Generator, Hashable
from tqdm import trange, tqdm
from datetime import datetime
from copy import deepcopy

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
if not os.path.exists("data/images"):
    os.makedirs("data/images", exist_ok=True)


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


def list_dir_visible(path: str):
    for i in os.listdir(path):
        if not i.startswith("."):
            yield i


def get_metric(experiment_name: str, metric_name: str) -> list:
    path = f"data/logs/automatic/{experiment_name}/metrics"
    latest = max(list_dir_visible(path))
    path = os.path.join(path, latest)
    metrics = Metrics({})
    metrics.load(path)
    # print(, "\n\n")
    # print(metrics.archived_metrics.keys())
    return metrics.archived_metrics[metric_name]


def get_multi_experiment_metric(experiment_base_name: str, metric_name: str) -> dict:
    path = f"data/logs/automatic/{experiment_base_name}"
    # print(path, list(list_dir_visible(path)))
    names = [(i, f"{experiment_base_name}/{i}") for i in list_dir_visible(path)]
    return {
        sub_name: get_metric(full_name, metric_name) for sub_name, full_name in names
    }


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


def run_test(dataset, model, loss_func, metrics):
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
    train_bar = tqdm(train_loader, desc=f"epoch {schedule.epoch}")
    optimizer = schedule.optimizer
    metrics.archive_metrics()
    for i, label in train_bar:
        optimizer.zero_grad()
        i = i.to(GLOBAL_DEVICE)
        label = label_converter(label)
        # print(i.mean(), i.std(), i.min(), i.max())
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
    # print(list(metrics.archived_metrics.keys()))
    metrics.recorded_values["loss"] = metrics.recorded_values.get("loss", []) + [
        output["loss"]
    ]
    # print(metrics.recorded_values["loss"])
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


def create_log_directory(test_name: str) -> None:
    log_dir = f"data/logs/automatic/{test_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir + "/metrics", exist_ok=True)
    os.makedirs(log_dir + "/images", exist_ok=True)


def experiment_from_config(config, verbose: bool = False) -> None:
    """
    assumes ffnn for now
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
    image_shape = config["image_shape"]
    n_pixels = image_shape[0] * image_shape[1]
    if config["mnist"]:
        dataset = get_MNIST(config["batch_size"], 256 * 4)
    else:
        dataset = get_pixel_shift(
            image_shape, (16384, 2048), config["batch_size"], 2048
        )
    model = FeedForwardVAE(
        n_pixels,
        config["latent_space_size"],
        config["width"],
        config["depth"],
        in_channels=1,
    ).to(GLOBAL_DEVICE)
    optimizer = config["optimizer"](model.parameters())
    schedule = Schedule(number_of_epochs=config["n_epochs"], optimizer=optimizer)
    schedule.manual_schedule(config["schedule"])
    loss_func = config["loss_func"]
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
    name = config["experiment_name"]
    create_log_directory(name)
    metric_values = run_experiment(dataset, schedule, model, loss_func, metrics)
    metrics.archive_metrics()
    metrics.dump(
        f"data/logs/automatic/{name}/metrics/{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        str(config),
    )
    # print(metric_values)
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
    # test_performance_line({"mig": metric_values["MIG"]})
    example_images = [dataset.test_dataset[i][0].to(GLOBAL_DEVICE) for i in range(10)]
    vae_visual_appraisal(
        model,
        name,
        example_images,
        GLOBAL_DEVICE,
        show=verbose,
    )


def direct_test():
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
            "lr": scale_dict_values({0: 3e-4}, 1e0),
        }
    )
    loss_func = ELBOLoss(0.0)
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
    create_log_directory("")
    metric_values = run_experiment(dataset, schedule, model, loss_func, metrics)

    print(metric_values)
    # visualisation
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


def main():
    config = {
        "experiment_name": "beta-test2",
        "image_shape": (28, 28),
        "batch_size": 256 + 128,
        "mnist": True,
        "width": 512,
        "depth": 3,
        "latent_space_size": 100,
        "optimizer": optim.AdamW,
        "schedule": {"lr": {0: 1e-4}},
        "n_epochs": 15,
        "loss_func": ELBOLoss(0.0),
    }
    config_adjustments = {
        "experiment_name": [
            "updated/b=0",
            # "beta-test2/b=1e-17",
            # "beta-test2/b=1e-15",
            # "beta-test2/b=1e-13",
            # "beta-test2/b=1e-5",
            # "beta-test2/b=1",
            # "beta-test2/b=10",
            # "beta-test2/b=100",
        ],
        "loss_func": [
            ELBOLoss(0),
            ELBOLoss(1e-17),
            ELBOLoss(1e-15),
            ELBOLoss(1e-13),
            # ELBOLoss(1e-5),
            # ELBOLoss(1),
            # ELBOLoss(10),
            # ELBOLoss(100),
        ],
    }
    configs = vary_config_variable(
        config,
        config_adjustments,
    )

    for config in configs:
        experiment_from_config(config, True)

    exit()
    test_performance_line(
        get_multi_experiment_metric("beta-test2", "loss"), title="ELBO loss"
    )
    test_performance_line(
        get_multi_experiment_metric("beta-test", "kl_div"), title="ELBO KL divergence"
    )
    test_performance_line(
        get_multi_experiment_metric("beta-test", "mse"), title="MSE loss"
    )
    test_performance_line(
        get_multi_experiment_metric("beta-test", "bce"), title="BCE loss"
    )


if "__main__" in __name__:
    main()
