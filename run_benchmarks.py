"""Benchmark and plot the different algorithms.
"""

import os
import time

import numpy as np
from numpy.random import RandomState

from mab import machines, solvers, viz


DUMP_DIR = './dump/'
TIME_STEPS = 5000
SEED_ITERS = 50
ALGOS = {
    'thompson-sampling': [solvers.ThompsonSampler, [0, 0]],
    'epsilon-greedy-0.1': [solvers.EpsilonGreedySampler, [0.01, 1.]],
    'epsilon-greedy-0.2': [solvers.EpsilonGreedySampler, [0.02, 1.]],
    'epsilon-greedy-0.3': [solvers.EpsilonGreedySampler, [0.03, 1.]],
    'epsilon-greedy-0.4': [solvers.EpsilonGreedySampler, [0.04, 1.]],
    'random-sampling': [solvers.RandomSampler, []]
}


def main():
    if not os.path.exists(DUMP_DIR):
        os.makedirs(DUMP_DIR)
    probas = [0.027, 0.03, 0.028, 0.001, 0.05, 0.06, 0.0234, 0.035, 0.01, 0.012]
    for i in range(SEED_ITERS):
        print("Iter: {}/{}".format(i+1, SEED_ITERS))
        seed = int(time.time())
        for name, (algo, args) in ALGOS.items():
            print('\tBenchmarking {}...'.format(name))
            rng = RandomState(seed)
            sm = machines.BernouilliSM(len(probas), probas, rng)
            solver = algo(sm, *args)
            solver.solve(TIME_STEPS)
            solver.save(DUMP_DIR, i)
        time.sleep(1)
    vis = viz.Visualizer(DUMP_DIR)
    vis.plot(save=True)


if __name__ == '__main__':
    main()
