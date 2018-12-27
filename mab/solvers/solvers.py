import numpy as np

from .base import Solver


class RandomSampler(Solver):
    """A sampler that randomly chooses what arm to pull.

    This should really only be used for baseline comparisons.
    """
    def __init__(self, sm):
        super().__init__(sm)

        self._reset()

    def __repr__(self):
        return "rand"

    def _reset(self):
        super()._reset()

    def _select_arm(self):
        return self.sm.rng.randint(self.sm.n)

    def _update(self, rew, idx):
        pass

    def _solve(self, t_i):
        idx = self._select_arm()
        r = self.sm.pull(idx)
        if r > 0. and r < 1.:
            rew = self.sm.rng.binomial(1, r)
        else:
            rew = r
        reg = self.sm.max_mean - self.sm.means[idx]
        self._update(rew, idx)
        return (idx, rew, reg)


class ThompsonSampler(Solver):
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
        return "ts-{}_{}".format(self._s_init, self._f_init)

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

    def _update(self, rew, idx):
        self._successes[idx] += rew
        self._failures[idx] += (1 - rew)

    def _solve(self, t_i):
        idx = self._select_arm()
        r = self.sm.pull(idx)
        if r > 0. and r < 1.:
            rew = self.sm.rng.binomial(1, r)
        else:
            rew = r
        reg = self.sm.max_mean - self.sm.means[idx]
        self._update(rew, idx)
        return (idx, rew, reg)


class EpsilonGreedySampler(Solver):
    """The ε-greedy algorithm.

    Attributes:
        eps: (float) the probability of randomly selecting
            an arm to pull. If 0, we perform greedy selection
            (i.e. no exploration).
        prob_init: (float) initial probability estimate for
            each arm.
        exp_iters: (float) how many iterations to spend on
            pure exploration.
    """
    def __init__(self, sm, eps, prob_init, exp_iters=0):
        self._eps = np.clip(eps, 0., 1.)
        self._prob_init = prob_init
        self._exp_iters = exp_iters
        self._explore = True

        super().__init__(sm)

        self._reset()

    def __repr__(self):
        return "eg-{}_{}".format(self._eps, self._prob_init)

    def _reset(self):
        super()._reset()
        self._probas = [self._prob_init] * self.sm.n

    def _select_arm(self):
        if self._explore or self.sm.rng.binomial(1, self._eps):
            return self.sm.rng.randint(0, self.sm.n)
        return np.argmax(self._probas)

    def _update(self, rew, idx):
        new_counter = self._counter[idx] + 1
        weighted_sum_1 = ((new_counter - 1) / (new_counter)) * self._probas[idx]
        weighted_sum_2 = (1. / (new_counter)) * rew
        self._probas[idx] = weighted_sum_1 + weighted_sum_2

    def _solve(self, t_i):
        if t_i >= self._exp_iters:
            self._explore = False
        idx = self._select_arm()
        r = self.sm.pull(idx)
        if r > 0. and r < 1.:
            rew = self.sm.rng.binomial(1, r)
        else:
            rew = r
        reg = self.sm.max_mean - self.sm.means[idx]
        self._update(rew, idx)
        return (idx, rew, reg)


class AnnealedEpsilonGreedySampler(Solver):
    pass
