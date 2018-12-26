"""Benchmark and plot the different algorithms.
"""

from mab import machines, solvers, viz


ALGOS = {
    'thompson-sampling': solvers.ThompsonSampling,
    'epsilon-greedy': solvers.EpsilonGreedy,
}


def main():
    vis = viz.Visualizer('/Users/kevin/Desktop/exp-1/')
    vis.plot_rewards()
    vis.plot_cumrewards()


if __name__ == '__main__':
    main()
