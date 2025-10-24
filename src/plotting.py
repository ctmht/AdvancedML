from itertools import product
from tkinter import W
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from torch import tensor, Tensor
import torch


def test_performance_line(
    test_losses: list[float] | dict[str, list[float]], log: bool = True
) -> None:
    """
    test performance plot for debugging and quick evaluation
    """
    if not isinstance(test_losses, dict):
        test_losses = {"loss": test_losses}
    fig = plt.figure(figsize=(8, 4.5))
    for name, line in test_losses.items():
        plt.plot(range(1, len(line) + 1), line, label=name)

    plt.legend()
    if log:
        plt.yscale("log")
    plt.show()


def show_image_grid(
    images: list,
    shape: tuple[int, int],
    vmax: int | float | None = 1.0,
    path: str | None = None,
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

    plt.show()
    if path:
        fig.savefig(path, format="pdf")


def vae_visual_appraisal(
    model, task_name, example_images: list[Tensor] | None = None, device=None
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
        f"data/images/{task_name}_latent_grid.pdf",
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
        f"data/images/{task_name}_examples_predicted.pdf",
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
        f"data/images/{task_name}_examples_predicted_normalized.pdf",
    )
    # show_image_grid(
    #     [model.generate(device=device) for _ in range(64)],
    #     (8, 8),
    #     f"data/images/{task_name}_generated_grid.pdf",
    # )
