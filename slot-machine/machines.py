import numpy as np

from .base import SlotMachine


class BernouilliSM(SlotMachine):
    """A slot machine with Bernouilli rewards.
    """
    def __init__(self, N, means, seed):
        super().__init__(N, seed)

        self.means = means

    def pull(self, i):
        """Uniformly samples a number in the range [0, 1].

        If the number is below the true mean of the arm,
        return 1. Else return 0.
        """
        if np.random.random() < self.means[i]:
            return 1.
        return 0.


class GaussianSM(SlotMachine):
    """A slot machine with Gaussian rewards.
    """
    def __init__(self, N, means, stds, seed):
        super().__init__(N, seed)

        self.means = means
        self.stds = stds

    def pull(self, i):
        """Samples a reward from a Gaussian distribution.

        The Gaussian is parameterized by `means[i]` and `stds[i]`.
        """
        return np.random.normal(self.means[i], self.stds[i])


class BinomialSM(SlotMachine):
    """A slot machine with Binomial rewards.
    """
    pass
