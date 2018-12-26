import numpy as np
from scipy.stats import truncnorm

from .base import SlotMachine


EPS = 1e-10  # for numerical stability


class BernouilliSM(SlotMachine):
    """A slot machine with Bernouilli rewards.
    """
    def __init__(self, n, means, rng):
        super().__init__(n, rng)

        self.means = means

    def pull(self, i):
        """Uniformly samples a number in the range [0, 1].

        If the number is below the true mean of the arm,
        return 1. Else return 0.
        """
        return self.rng.binomial(1, self.means[i])

    @property
    def max_mean(self):
        """Returns the max arm mean.
        """
        return np.max(self.means)


class GaussianSM(SlotMachine):
    """A slot machine with Gaussian rewards.
    """
    EPS = 1e-10  # for numerical stability

    def __init__(self, n, means, stds, rng):
        super().__init__(n, rng)

        self.means = means
        self.stds = stds

    def pull(self, i):
        """Samples a reward from a truncated Gaussian distribution.

        The Gaussian is parameterized by `means[i]` and `stds[i]`.
        """
        return truncnorm.rvs(
            (0 - self.means[i]) / (self.stds[i] + EPS),
            (1 - self.means[i]) / (self.stds[i] + EPS),
            loc=self.means[i],
            scale=self.stds[i],
            seed=self.rng,
        )

    @property
    def max_mean(self):
        """Returns the max arm mean.
        """
        return np.max(self.means)


class BinomialSM(SlotMachine):
    """A slot machine with Binomial rewards.
    """
    pass
