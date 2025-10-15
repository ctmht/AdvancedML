import torch
from torch import Tensor
from torch.distributions import Distribution
from disent.metrics._mig import _compute_mig


class VAEMetric:
    def __call__(
        self,
        data_example: Tensor,
        latent_dist: Distribution,
        output: Tensor,
        factors: Tensor,
    ) -> float:
        pass


class Metrics:
    """used to combine metrics for easy use"""

    def __init__(self, metrics_used: dict) -> None:
        self.metrics_used = metrics_used
        self.recorded_values = {}
        self.reset_metrics()

    def reset_metrics(self) -> None:
        self.recorded_values = {i: [] for i in self.metrics_used.keys()}

    def __call__(
        self,
        data_example: Tensor,
        latent_dist: Distribution,
        output: Tensor,
        factors: Tensor,
    ) -> None:
        for name, metric in self.metrics_used.items():
            self.recorded_values[name].append(
                metric(data_example, latent_dist, output, factors)
            )

    def mean_metrics(self) -> dict:
        return {i: sum(v) / len(v) for i, v in self.recorded_values.items()}


class ELBOLoss(VAEMetric):
    def __init__(self, beta: float = 1.0) -> None:
        self.beta = beta

    def __call__(self, data_example, latent_dist, output, factors) -> float:
        mu, var = latent_dist.mean, latent_dist.variance
        kl_div = (mu**2 + var - 1 - torch.log(var)).sum(dim=1).mean()
        exp_loss = 0.5 * ((data_example - output) ** 2).mean()
        return exp_loss + self.beta * kl_div


class MIG(VAEMetric):
    def __call__(self, data_example, latent_vector, output, factors) -> float:
        # print(latent_vector.mean.shape, factors.shape)
        return _compute_mig(
            latent_vector.mean.detach().cpu().numpy().T,
            factors.detach().cpu().numpy().T,
        )["mig.discrete_score"]
