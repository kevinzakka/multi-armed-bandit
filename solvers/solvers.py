import numpy as np

from .base import Solver


class ThompsonSampling(Solver):
    """The Thompson Sampling algorithm.

    References:
        http://proceedings.mlr.press/v23/agrawal12/agrawal12.pdf
    """
    def __init__(self):
        super().__init__()
        self._a = 0
        self._b = 0

    def _update(self, success):
        if success:
            self._a += 1
        else:
            self._b += 1

    def _increment_both(self):
        self._a += 1
        self._b += 1

    def sample(self, shape=None):
        self._increment_both()
        return np.random.beta(self._a, self._b)
