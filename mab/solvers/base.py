"""Various algorithms for solving stochastic multi-armed-bandit problems.
"""

from abc import ABC, abstractmethod

import numpy as np


class Solver(ABC):
    """A stochastic multi-armed-bandit solver.

    Attributes:
        sm: (SlotMachine) a slot machine we would like
            to play for a fixed number of time steps `t`. The
            goal is to maximize the expected total reward in time `t`.
    """
    @abstractmethod
    def __init__(self, sm):
        self.sm = sm

        self._reset()

    @abstractmethod
    def _reset(self):
        """Resets the solver.
        """
        self._counts = [0.] * self.sm.n  # number of times each arm has been pulled
        self._rewards = []  # history of rewards
        np.random.seed(self.sm.seed)

    @abstractmethod
    def _select_arm(self):
        """Selects the arm to pull at a given time step `t`.
        """
        pass

    @abstractmethod
    def _update(self, r, idx):
        """Update the solver's beliefs at each end of
        a single time step based on a received reward `r`
        and selected arm `idx`.
        """
        pass

    @abstractmethod
    def _solve(self):
        """Runs the solver for one time step.
        """
        pass

    def solve(self, t):
        """Runs the solver for a fixed number of time steps `t`.
        """
        self._reset()
        for _ in range(t):
            idx, r = self._solve()  # run for a single time step
            self._counts[idx] += 1  # keep track of what arm was picked
            self._rewards.append(r)  # store obtained reward

    @property
    def counts(self):
        return self._counts

    @property
    def rewards(self):
        return self._rewards

    @property
    def regrets(self):
        return np.cumsum(self._regrets)

    @property
    def cumulative_rewards(self):
        return np.cumsum(self._rewards)
