"""Slot machines with various reward distributions.
"""

from abc import ABC, abstractmethod


class SlotMachine(ABC):
    """A slot machine with `n` arms.

    At every time step `t`, one of the `n` arms is chosen. When
    pulled, every arm `i` yields a random real-valued reward
    according to some fixed (unknown) distribution with support
    in [0, 1]. The random rewards obtained from pulling an arm
    repeatedly are i.i.d. and independent of the pulls of the
    other arms. The reward is observed immediately after pulling
    the arm.

    Attributes:
        n: (int) number of arms.
        rng: (RandomState) a numpy RNG.
    """
    @abstractmethod
    def __init__(self, n, rng):
        self.n = n
        self.rng = rng

    @abstractmethod
    def pull(self, i):
        """Pulls the i'th arm and returns the observed reward.
        """
        pass
