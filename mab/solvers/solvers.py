import numpy as np

from .base import Solver


class ThompsonSampling(Solver):
    """The Thompson Sampling algorithm.

    Attributes:
        s_init: (int) initial value for the `alpha` parameter of
            the Beta distribution. Assumed the same for all arms.
        f_init: (int) initial value for the `beta` parameter of
            the Beta distribution. Assumed the same for all arms.

    References:
        http://proceedings.mlr.press/v23/agrawal12/agrawal12.pdf
    """
    def __init__(self, sm, s_init, f_init):
        self._s_init = int(s_init)
        self._f_init = int(f_init)

        super().__init__(sm)

        self._reset()

    def __repr__(self):
        return "ts"

    def _reset(self):
        super()._reset()
        self._successes = [self._s_init] * self.sm.n
        self._failures = [self._f_init] * self.sm.n

    def _select_arm(self):
        thetas = []
        for i in range(self.sm.n):
            thetas.append(
                self.sm.rng.beta(
                    self._successes[i]+1,
                    self._failures[i]+1,
                )
            )
        return np.argmax(thetas)

    def _update(self, r, idx):
        self._successes[idx] += r
        self._failures[idx] += (1 - r)

    def _solve(self):
        # decide what arm to pull
        idx = self._select_arm()

        # pull arm and receive reward
        r_ = self.sm.pull(idx)

        # if continuous, make binary
        if r_ > 0. and r_ < 1.:
            r = self.sm.rng.binomial(1, r_)
        else:
            r = r_

        # update success and failure counts
        self._update(r, idx)

        return (idx, r)


class EpsilonGreedy(Solver):
    """The ε-greedy algorithm.

    Attributes:
        eps: (float) the probability of randomly selecting
            an arm to pull. If 0, we perform greedy selection
            (i.e. no exploration).
        prob_init: (float) initial probability estimate for
            each arm.
    """
    def __init__(self, sm, eps, prob_init):
        self._eps = np.clip(eps, 0., 1.)
        self._prob_init = prob_init
        self._explore = False

        super().__init__(sm)

        self._reset()

    def __repr__(self):
        return "eg"

    def _reset(self):
        super()._reset()
        self._probas = [self._prob_init] * self.sm.n

    def _select_arm(self):
        # decide whether to explore
        self._explore = self.sm.rng.binomial(1, self._eps)

        if self._explore:
            # select an arm with uniform probability
            return self.sm.rng.randint(0, self.sm.n)
        else:
            # select the arm with the highest probability
            return np.argmax(self._probas)

    def _update(self, r, idx):
        new_counter = self._counter[idx] + 1
        weighted_sum_1 = ((new_counter - 1) / (new_counter)) * self._probas[idx]
        weighted_sum_2 = (1. / (new_counter)) * r
        self._probas[idx] = weighted_sum_1 + weighted_sum_2

    def _solve(self):
        # decide what arm to pull
        idx = self._select_arm()

        # pull arm and receive reward
        r_ = self.sm.pull(idx)

        # if continuous, make binary
        if r_ > 0. and r_ < 1.:
            r = self.sm.rng.binomial(1, r_)
        else:
            r = r_

        # update probability estimates
        self._update(r, idx)

        return (idx, r)


class AnnealedEpsilonGreedy(Solver):
    pass
