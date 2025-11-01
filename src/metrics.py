from itertools import product
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

    def __init__(
        self, metrics_used: dict, prefix_options: list[str] | None = None
    ) -> None:
        prefix_options = prefix_options or [""]
        self.metrics = metrics_used
        self.metrics_used = {
            i + k: v for i, (k, v) in product(prefix_options, metrics_used.items())
        }
        self.recorded_values = {}
        self.archived_metrics = {i: [] for i in self.metrics_used.keys()}
        self.reset_metrics()

    def reset_metrics(self) -> None:
        self.recorded_values = {i: [] for i in self.metrics_used.keys()}

    def archive_metrics(self) -> None:
        for k, v in self.recorded_values.items():
            if k not in self.archived_metrics:
                self.archived_metrics[k] = []
            self.archived_metrics[k] += v
            self.recorded_values[k] = []

    def __call__(
        self,
        data_example: Tensor,
        latent_dist: Distribution,
        output: Tensor,
        factors: Tensor,
        prefix: str = "",
        exclude: list[str] | None = None,
    ) -> None:
        exclude = exclude or []
        for name, metric in self.metrics.items():
            if name in exclude:
                continue
            metric_value = metric(data_example, latent_dist, output, factors)
            if isinstance(metric_value, Tensor):
                metric_value = metric_value.item()

            self.recorded_values[prefix + name].append(metric_value)

    def mean_metrics(self) -> dict:
        return {i: sum(v) / len(v) for i, v in self.recorded_values.items() if v}

    def dump(self, path: str, description: str = "") -> None:
        pickle.dump(
            (description, self.recorded_values, self.archived_metrics), open(path, "wb")
        )

    def load(self, path: str) -> str:
        """returns description"""
        d, v, a = pickle.load(open(path, "rb"))
        self.recorded_values = v
        self.archived_metrics = a
        return d


class ELBOLoss(VAEMetric):
    def __init__(self, beta: float = 1.0) -> None:
        self.beta = beta
        self.exp_loss = MSE()
        self.kl_div = GaussKLdiv()

    def __repr__(self) -> str:
        return f"ELBO loss, beta = {self.beta}, reconstruction loss is {type(self.exp_loss)}"

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
        return ((data_example - output) ** 2).mean(dim=0).sum()


class LatentMean(VAEMetric):
    def __call__(
        self,
        data_example: Tensor,
        latent_dist: Distribution,
        output: Tensor,
        factors: Tensor,
    ) -> Tensor | float:
        return latent_dist.mean.mean()


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
        self.bce = nn.BCELoss(reduction="sum")

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
        # raise NotImplementedError("MIG temporarily disabled")
        return _compute_mig(
            latent_vector.mean.detach().cpu().numpy().T,
            factors.detach().cpu().numpy().T,
        )["mig.discrete_score"]
