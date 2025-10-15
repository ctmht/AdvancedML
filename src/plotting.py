from itertools import product
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from torch import tensor, Tensor
import torch


def test_performance_line(test_losses: list[float] | list[list[float]]) -> None:
    """
    test performance plot for debugging and quick evaluation
    """
    if not isinstance(test_losses[0], list):
        test_losses = [test_losses]
    fig = plt.figure(figsize=(8, 4.5))
    for i in test_losses:
        plt.plot(range(1, len(i) + 1), i)
    plt.yscale("log")
    plt.show()


def show_image_grid(
    images: list,
    shape: tuple[int, int],
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
        ax.imshow(im)

    plt.show()
    if path:
        fig.savefig(path, format="pdf")


def vae_visual_appraisal(
    model, task_name, example_images: list[Tensor] | None = None, device=None
):
    """
    No, you don't want to take a close look at this function.
    """
    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
        )
    show_image_grid(
        [
            model.generate(tensor([a, b] + [0] * (model.latent_size - 2)).to(device))
            for a, b in product(
                [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0], repeat=2
            )
        ],
        (8, 8),
        f"data/images/{task_name}_latent_grid.pdf",
    )
    if example_images is None:
        return
    show_image_grid(
        [i.cpu()[0] for i in example_images]
        + [
            model(i.view(1, -1, 28, 28))[0][0, 0].detach().cpu().numpy()
            for i in example_images
        ],
        (2, 10),
        f"data/images/{task_name}_examples_predicted.pdf",
    )
    show_image_grid(
        [model.generate() for i in range(64)],
        (8, 8),
        f"data/images/{task_name}_generated_grid.pdf",
    )
