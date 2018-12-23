"""Various algorithms for solving stochastic multi-armed-bandit problems.
"""

import numpy as np

from abc import ABC, abstractmethod


class Solver(ABC):
    """A stochastic MAB solver.

    Attributes:
        sm: (SlotMachine) a slot machine we would like
            to play for a fixed number of time steps `T`. The
            goal is to maximize the expected total reward in time `T`.
    """
    @abstractmethod
    def __init__(self, sm):
        self.sm = sm

    @abstractmethod
    def step(self, i):
        pass
