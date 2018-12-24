"""Various algorithms for solving stochastic multi-armed-bandit problems.
"""

from abc import ABC, abstractmethod

import numpy as np
from ipdb import set_trace as debug

class Solver(ABC):
    """A stochastic multi-armed-bandit solver.

    Attributes:
        sm: (SlotMachine) a slot machine we would like
            to play for a fixed number of time steps `T`. The
            goal is to maximize the expected total reward in time `T`.
    """
    @abstractmethod
    def __init__(self, sm):
        self.sm = sm

        self._reset()

    @abstractmethod
    def _reset(self):
        """Resets the solver.
        """
        self.counts = [0.] * self.sm.N  # number of times each arm has been pulled
        self.rewards = []  # history of rewards
        np.random.seed(self.sm.seed)

    @abstractmethod
    def _select_arm(self):
        """Selects the arm to pull at a given time step `t`.
        """
        pass

    @abstractmethod
    def _update(self, r, idx):
        """Update the solver's beliefs at the end of a
        given time step `t` based on a received reward `r`
        and select arm `idx`.
        """
        pass

    @abstractmethod
    def _solve(self):
        """Runs the solver for a single time step `t`.
        """
        pass

    def solve(self, T):
        """Runs the solver for a fixed number of time steps `T`.
        """
        for t in range(T):
            idx, r = self._solve()
            self.counts[idx] += 1
            self.rewards.append(r)
