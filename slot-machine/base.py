"""Slot machines with different rewards.
"""

import numpy as np

from abc import ABC, abstractmethod


class SlotMachine(ABC):
    """A slot machine with `N` arms.

    At every time step `t`, one of the N arms is chosen. When
    pulled, every arm `i` yields a random real-valued reward
    according to some fixed (unknown) distribution with support
    in [0, 1]. The random rewards obtained from pulling an arm
    repeatedly are i.i.d. and independent of the pulls of the
    other arms. The reward is observed immediately after pulling
    the arm.

    Attributes:
        N: (int) number of arms.
        seed: (float) RNG seed.
    """
    @abstractmethod
    def __init__(self, N, seed):
        np.random.seed(seed)

        self.N = N

    @abstractmethod
    def pull(self, i):
        """Pulls the i'th arm and returns the observed reward.
        """
        pass
