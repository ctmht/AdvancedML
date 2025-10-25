import pickle
import torch
import torch.nn as nn
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
    ) -> Tensor | float:
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
            metric_value = metric(data_example, latent_dist, output, factors)
            if isinstance(metric_value, Tensor):
                metric_value = metric_value.item()

            self.recorded_values[name].append(metric_value)

    def mean_metrics(self) -> dict:
        return {i: sum(v) / len(v) for i, v in self.recorded_values.items()}

    def dump(self, path: str, description: str = "") -> None:
        pickle.dump((description, self.recorded_values), open(path, "wb"))

    def load(self, path: str) -> str:
        """returns description"""
        d, v = pickle.load(open(path, "rb"))
        self.recorded_values = v
        return d


class ELBOLoss(VAEMetric):
    def __init__(self, beta: float = 1.0) -> None:
        self.beta = beta
        self.exp_loss = BinaryCrossExtropy()
        self.kl_div = GaussKLdiv()

    def __call__(self, data_example, latent_dist, output, factors) -> Tensor:
        exp_loss = self.exp_loss(data_example, latent_dist, output, factors)
        kl_div = self.kl_div(data_example, latent_dist, output, factors)
        return exp_loss + self.beta * kl_div


class MSE(VAEMetric):
    def __call__(
        self,
        data_example: Tensor,
        latent_dist: Distribution,
        output: Tensor,
        factors: Tensor,
    ) -> Tensor:
        return ((data_example - output) ** 2).mean()


class LatentMean(VAEMetric):
    def __call__(
        self,
        data_example: Tensor,
        latent_dist: Distribution,
        output: Tensor,
        factors: Tensor,
    ) -> Tensor | float:
        return latent_dist.mean.sum()


class LatentStddev(VAEMetric):
    def __call__(
        self,
        data_example: Tensor,
        latent_dist: Distribution,
        output: Tensor,
        factors: Tensor,
    ) -> Tensor | float:
        return latent_dist.stddev.mean()


class BinaryCrossExtropy(VAEMetric):
    def __init__(self) -> None:
        self.bce = nn.BCELoss()

    def __call__(
        self,
        data_example: Tensor,
        latent_dist: Distribution,
        output: Tensor,
        factors: Tensor,
    ) -> float:
        return self.bce(output.clamp(0, 1).nan_to_num(1), data_example.clamp(0, 1))


class GaussKLdiv(VAEMetric):
    def __call__(
        self,
        data_example: Tensor,
        latent_dist: Distribution,
        output: Tensor,
        factors: Tensor,
    ) -> Tensor:
        mu, var = latent_dist.mean, latent_dist.variance
        kl_div = (mu**2 + var - 1 - torch.log(var)).sum(dim=1).mean()
        return kl_div


class MIG(VAEMetric):
    def __call__(self, data_example, latent_vector, output, factors) -> float:
        # print(latent_vector.mean.shape, factors.shape)
        return _compute_mig(
            latent_vector.mean.detach().cpu().numpy().T,
            factors.detach().cpu().numpy().T,
        )["mig.discrete_score"]
