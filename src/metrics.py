import torch
from torch import Tensor
from torch.distributions import Distribution


class VAEMetric:
    def __call__(
        self, data_example: Tensor, latent_dist: Distribution, output: Tensor
    ) -> float:
        pass


class ELBOLoss(VAEMetric):
    def __init__(self, beta: float = 1.0) -> None:
        self.beta = beta

    def __call__(self, data_example, latent_dist, output) -> float:
        mu, var = latent_dist.mean, latent_dist.variance
        kl_div = (mu**2 + var - 1 - torch.log(var)).sum(dim=1).mean()
        exp_loss = 0.5 * ((data_example - output) ** 2).mean()
        return exp_loss + self.beta * kl_div


class MIG(VAEMetric):
    def __init__(self) -> None:
        pass

    def __call__(self, data_example, latent_vector, output) -> float:
        pass
