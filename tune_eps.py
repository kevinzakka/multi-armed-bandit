"""Tune the value of ε.
"""

import numpy as np
from matplotlib import pyplot as plt
from numpy.random import RandomState

from mab import machines, solvers


def main():
    probas = [0.027, 0.03, 0.028, 0.001, 0.05, 0.06, 0.0234, 0.035, 0.01, 0.012]
    epsilons = np.linspace(0, 1, 500)
    total_rewards = []
    for seed in [1, 2432412, 434, 439421094, 86868]:
        rewards = []
        for eps in epsilons:
            rng = RandomState(seed)
            sm = machines.BernouilliSM(len(probas), probas, rng)
            solver = solvers.EpsilonGreedySampler(sm, eps,  1.0)
            solver.solve(1000)
            rewards.append(np.sum(solver.rewards))
        total_rewards.append(rewards)
    total_rewards = np.mean(total_rewards, axis=0)
    best_eps = epsilons[np.argmax(total_rewards)]
    best_rew = np.max(total_rewards)
    print("Best ε: {}".format(best_eps))
    plt.figure(figsize=(15, 8))
    plt.plot(epsilons, total_rewards, '.-')
    plt.axvline(x=best_eps, color='r', linestyle='--', alpha=0.4)
    plt.axhline(y=best_rew, color='r', linestyle='--', alpha=0.4)
    plt.plot(best_eps, best_rew, 'r*')
    plt.xlabel("Value of ε")
    plt.ylabel("Total Reward")
    plt.grid('k', ls='--', alpha=0.3)
    plt.savefig("./assets/eps-tune.png", format="png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
