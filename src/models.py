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


class Identity(nn.Module):
    def __init__(self) -> None:
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class BaseVAE(nn.Module):
    def encode(self, x: Tensor) -> Tensor:
        encoder_outputs = self.encoder(x)
        return self.ff_mean(encoder_outputs), self.ff_variance(encoder_outputs)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x) -> Tensor:
        eps = 1e-10
        m, v = self.encode(x)
        z = dist.Normal(m, v**2 + eps)  # 2.72 ** (v))  # torch.exp(v))
        return self.decode(z.sample()), z

    def generate(
        self, z: Tensor | None = None, return_image: bool = True, device=None
    ) -> Tensor:
        s = self.expected_latent_dist.sample()
        z = (s.to(device) if device is not None else s) if z is None else z
        if return_image:
            # print(z.shape)
            # print(self.decode(z).shape)
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


class FeedForwardVAE(BaseVAE):
    def __init__(
        self,
        in_size: int,
        latent_size: int,
        hidden_size: int,
        depth: int = 2,
        in_channels: int = 3,
    ) -> None:
        super(FeedForwardVAE, self).__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.depth = depth
        if depth < 2:
            raise ValueError("Depth has to be >= 2")
        im_out_size = int(sqrt(in_size // in_channels))
        # encoder layers:
        self.encoder = nn.Sequential(
            *(
                [nn.Flatten(), self.layer_block(in_size, hidden_size)]
                + [self.layer_block(hidden_size, hidden_size) for i in range(depth - 2)]
            )
        )
        self.ff_mean = nn.Linear(hidden_size, latent_size)
        self.ff_variance = nn.Linear(hidden_size, latent_size)
        self.decoder = nn.Sequential(
            *(
                [View(-1, latent_size), self.layer_block(latent_size, hidden_size)]
                + [self.layer_block(hidden_size, hidden_size) for _ in range(depth - 2)]
                + [
                    self.layer_block(hidden_size, in_size),
                    View(-1, in_channels, im_out_size, im_out_size),
                ]
            )
        )
        self.expected_latent_dist = dist.Normal(
            torch.zeros(latent_size), torch.ones(latent_size)
        )

    def layer_block(
        self,
        in_size: int,
        out_size: int,
        batch_norm: bool = True,
        act_func: bool = True,
    ) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.LazyBatchNorm1d() if batch_norm else Identity(),
            nn.Sigmoid() if act_func else Identity(),
            # nn.LeakyReLU(0.1),
        )


class ResnetBlock(nn.Module):
    def __init__(self, channels: int = 128) -> None:
        super(ResnetBlock, self).__init__()
        self.internal_module = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), 1, (1, 1)),
            nn.MaxPool2d((2, 2), 1, (1, 1)),
            nn.LazyBatchNorm2d(),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channels, channels, (3, 3), 1, (1, 1)),
            nn.MaxPool2d((2, 2), 1, (1, 1)),
        )
        self.final_module = nn.Sequential(nn.LazyBatchNorm2d(), nn.LeakyReLU(0.1))

    def forward(self, x: Tensor) -> Tensor:
        return self.final_module(self.internal_module(x.clone()) + x)


class ResNet:
    def __init__(self, channels: int, depth: int) -> None:
        super(ResNet, self).__init__()

        self.residual_blocks = nn.Sequential(
            *(ResnetBlock(channels) for _ in range(depth))
        )


class VQVAE(BaseVAE):
    def __init__(self) -> None:
        super(VQVAE, self).__init__()

    def forward(self, x):
        pass
