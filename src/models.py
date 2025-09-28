import torch.nn as nn
import torch.distributions as dist
from torch import Tensor
import torch
from math import sqrt


# Too lazy to code myself, so I copied it. Weird that it isn't built in.
# https://discuss.pytorch.org/t/how-to-build-a-view-layer-in-pytorch-for-sequential-models/53958/3
class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class BaseVAE(nn.Module):
    def encode(self, x: Tensor) -> Tensor:
        encoder_outputs = self.encoder(x)
        return self.ff_mean(encoder_outputs), self.ff_variance(encoder_outputs).abs()

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x) -> Tensor:
        eps = 1e-10
        m, v = self.encode(x)
        z = dist.Normal(m, v + eps)
        return self.decode(z.sample()), z

    def generate(self, z: Tensor | None = None, return_image: bool = True) -> Tensor:
        z = self.expected_latent_dist.sample() if z is None else z
        if return_image:
            return self.decode(z)[0, 0].detach().cpu().numpy()
        return self.decode(z), z

    def get_latent_size(self) -> int:
        return self.latent_size  # models must have a parameter latent_size!


class ExampleVAE(BaseVAE):
    def __init__(
        self,
        in_size: int,
        latent_size: int,
        in_channels: int = 3,
        inter_channels: tuple[int, int] | None = None,
    ) -> None:
        super(ExampleVAE, self).__init__()
        self.latent_size = latent_size
        first_channels = inter_channels[0] if inter_channels else 8
        mid_channels = inter_channels[1] if inter_channels else 16
        out_size = mid_channels * in_size
        im_out_size = int(sqrt(out_size // mid_channels))
        # encoder layers:
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, first_channels, 3, 1, padding=1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.Conv2d(first_channels, mid_channels, 3, 1, padding=1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.ff_mean = nn.Linear(out_size, latent_size)
        self.ff_variance = nn.Linear(out_size, latent_size)
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, out_size),
            View(-1, mid_channels, im_out_size, im_out_size),
            nn.ConvTranspose2d(mid_channels, first_channels, 3, 1, padding=1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.ConvTranspose2d(first_channels, in_channels, 3, 1, padding=1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
        )
        self.expected_latent_dist = dist.Normal(
            torch.zeros(latent_size), torch.ones(latent_size)
        )
