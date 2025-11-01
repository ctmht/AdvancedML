from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from torch import tensor, Tensor
import torch
import seaborn as sns
import os

from metrics import Metrics

sns.set_theme()


def mean(iterable):
    return sum(iterable) / len(iterable)


def test_performance_line(
    test_losses: list[float] | dict[str, list[float]],
    log: bool = True,
    path: str | None = None,
    show: bool = True,
    title: str = "",
) -> None:
    """
    test performance plot for debugging and quick evaluation
    """
    if not isinstance(test_losses, dict):
        test_losses = {"loss": test_losses}
    fig = plt.figure(figsize=(8, 4.5))
    for name, line in test_losses.items():
        plt.plot(range(1, len(line) + 1), line, label=name)

    plt.title(title)
    plt.legend()
    if log:
        plt.yscale("log")
    if path is not None:
        fig.savefig(f"data/logs/automatic/{path}")
    if show:
        plt.show()
    plt.close("all")


def variable_value_lines(
    data: list[float] | dict[str, list[float]],
    log: bool = True,
    path: str | None = None,
    show: bool = True,
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    order: list[str] | None = None,
) -> None:
    """
    Create a prettier line plot for checking the changes in \
    values across different values of a variable. This allows \
    the use of colour gradients. 
    """
    if not isinstance(data, dict):
        data = {"loss": data}
    scaling = 0.6
    fig = plt.figure(figsize=(16 * scaling, 9 * scaling))
    colours = sns.color_palette("crest", n_colors=len(data.keys()))
    names = order if order else data.keys()
    for name, c in zip(names, colours):
        line = data[name]
        plt.plot(
            range(1, len(line) + 1),
            line,
            label=name,
            color=c,
            linewidth=3,
        )

    plt.title(title)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if log:
        plt.yscale("log")
    if path is not None:
        fig.savefig(f"data/logs/manual/{path}.png", dpi=300)
    if show:
        plt.show()
    plt.close("all")


def show_image_grid(
    images: list,
    shape: tuple[int, int],
    vmax: int | float | None = 1.0,
    path: str | None = None,
    show: bool = True,
) -> None:
    fig = plt.figure(figsize=(16.0, 9.0))
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=shape,
        axes_pad=0.0,
    )

    for ax, im in zip(grid, images):
        ax.imshow(im, vmax=vmax)

    if path:
        fig.savefig(path, format="pdf")
    if show:
        plt.show()
    plt.close("all")


def vae_visual_appraisal(
    model,
    task_name,
    example_images: list[Tensor] | None = None,
    device=None,
    show: bool = True,
):
    """
    No, you don't want to take a close look at this function.
    """
    model.eval()
    value_range = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    value_range = [5 * i for i in value_range]
    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
        )
    image_grid = [
        model.generate(
            tensor([a, b] + [0] * (model.latent_size - 2)).view(1, 1, -1).to(device)
        )
        for a, b in product(value_range, repeat=2)
    ]
    show_image_grid(
        image_grid,
        (8, 8),
        1,
        f"data/logs/automatic/{task_name}/images/latent_grid.pdf",
        show,
    )
    if example_images is None:
        return
    image_size = image_grid[0].shape[-2:]
    show_image_grid(
        [i.cpu()[0] for i in example_images]
        + [
            model(i.view(1, -1, image_size[0], image_size[1]).to(device))[0][0, 0]
            .detach()
            .cpu()
            .numpy()
            for i in example_images
        ],
        (2, 10),
        1,
        f"data/logs/automatic/{task_name}/images/examples_predicted.pdf",
        show,
    )
    show_image_grid(
        [i.cpu()[0] for i in example_images]
        + [
            model(i.view(1, -1, image_size[0], image_size[1]).to(device))[0][0, 0]
            .detach()
            .cpu()
            .numpy()
            for i in example_images
        ],
        (2, 10),
        None,
        f"data/logs/automatic/{task_name}/images/examples_predicted_normalized.pdf",
        show,
    )
    # show_image_grid(
    #     [model.generate(device=device) for _ in range(64)],
    #     (8, 8),
    #     f"data/images/{task_name}_generated_grid.pdf",
    # )


def list_dir_visible(path: str):
    """list all public files in a directory"""
    for i in os.listdir(path):
        if not i.startswith("."):
            yield i


def get_metric(experiment_name: str, metric_name: str) -> list:
    """Get a metric from an experiment log"""
    path = f"data/logs/automatic/{experiment_name}/metrics"
    latest = max(list_dir_visible(path))
    path = os.path.join(path, latest)
    metrics = Metrics({})
    metrics.load(path)
    return metrics.archived_metrics[metric_name]


def get_multi_experiment_metric(experiment_base_name: str, metric_name: str) -> dict:
    """
    Get the metric data for a given metric across all experiments in a multi-config experiment.
    """
    path = f"data/logs/automatic/{experiment_base_name}"
    # print(path, list(list_dir_visible(path)))
    names = [(i, f"{experiment_base_name}/{i}") for i in list_dir_visible(path)]
    return {
        sub_name: get_metric(full_name, metric_name) for sub_name, full_name in names
    }


def multi_experiment_plotting(base_name: str, mig_included: bool = True):
    """Plot several metrics across a multi-experiment base_name"""
    order = ["b=0", "b=0.1", "b=0.5", "b=1", "b=2", "b=5"]
    variable_value_lines(
        {
            k.split("z")[0]: v
            for k, v in get_multi_experiment_metric(base_name, "loss").items()
            if k.endswith("zs=4")
        },
        title="ELBO loss",
        x_label="epochs",
        y_label="ELBO loss",
        path="latest_loss",
        order=order,
    )
    variable_value_lines(
        {
            k.split("z")[0]: v
            for k, v in get_multi_experiment_metric(base_name, "kl_div").items()
            if k.endswith("zs=4")
        },
        title="ELBO KL divergence",
        x_label="epochs",
        y_label="KL divergence",
        path="latest_kl_div",
        order=order,
    )
    variable_value_lines(
        {
            k.split("z")[0]: v
            for k, v in get_multi_experiment_metric(base_name, "mse").items()
            if k.endswith("zs=4")
        },
        title="MSE loss",
        x_label="epochs",
        y_label="Mean Squared Error",
        path="latest_mse",
        order=order,
    )
    variable_value_lines(
        {
            k.split("z")[0]: v
            for k, v in get_multi_experiment_metric(base_name, "bce").items()
            if k.endswith("zs=4")
        },
        title="BCE loss",
        x_label="epochs",
        y_label="Binary Cross Entropy",
        path="latest_bce",
        order=order,
    )

    if mig_included:
        order = ["b=0", "b=0.1", "b=0.5", "b=0.75", "b=1", "b=1.25", "b=1.5", "b=2"]
        variable_value_lines(
            {
                k.split("z")[0]: [mean(i) for i in batched(v, 384)]
                for k, v in get_multi_experiment_metric(base_name, "MIG").items()
                if k.endswith("zs=10")
            },
            title="MIG metric for MNIST",
            x_label="epochs",
            y_label="Mutual Information Gap",
            path="latest_mig",
            order=order,
            log=False,
        )


if "__main__" in __name__:
    multi_experiment_plotting("MNIST_grid_test", False)
