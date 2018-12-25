"""Various algorithms for solving stochastic multi-armed-bandit problems.
"""

from abc import ABC, abstractmethod

import os
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
    def __repr__(self):
        pass

    @abstractmethod
    def _reset(self):
        """Resets the solver.
        """
        self._counter = [0.] * self.sm.n  # a counter for each arm's number of pulls
        self._rewards = []  # all rewards up to time step `t`.

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
        # clear all internal variables
        self._reset()

        for _ in range(t):
            # run for a single time step
            idx, r = self._solve()

            # keep track of what arm was pulled
            self._counter[idx] += 1

            # store obtained reward
            self._rewards.append(r)

    @property
    def counter(self):
        return self._counter

    @property
    def rewards(self):
        return self._rewards

    @property
    def cumreward(self):
        return np.cumsum(self._rewards)
