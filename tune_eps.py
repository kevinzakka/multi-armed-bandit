"""Tune the value of ε.
"""

import numpy as np
from matplotlib import pyplot as plt
from numpy.random import RandomState

from mab import machines, solvers


def main():
    probas = [np.random.random() for _ in range(50)]
    epsilons = np.linspace(0, 1, 10)
    total_rewards = []
    for seed in [1, 2432412, 434, 439421094, 86868]:
        rng = RandomState(seed)
        sm = machines.BernouilliSM(len(probas), probas, rng)
        rewards = []
        for eps in epsilons:
            solver = solvers.EpsilonGreedy(sm, eps,  1.0)
            solver.solve(1000)
            rewards.append(np.sum(solver.rewards))
        total_rewards.append(rewards)
    total_rewards = np.mean(total_rewards, axis=0)
    plt.plot(epsilons, total_rewards, '-o')
    plt.xlabel("Value of ε")
    plt.ylabel("Total Reward")
    plt.grid('k', ls='--', alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
