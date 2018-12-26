"""Benchmark and plot the different algorithms.
"""

import os
import time

import numpy as np
from numpy.random import RandomState

from mab import machines, solvers, viz


DUMP_DIR = './dump/'
TIME_STEPS = 500 # how many solver runs
SEED_ITERS = 2000 # how many times to run everything
ALGOS = {
    'thompson-sampling': [solvers.ThompsonSampling, [0, 0]],
    'epsilon-greedy-0.1': [solvers.EpsilonGreedy, [0.01, 1.]],
    'epsilon-greedy-0.2': [solvers.EpsilonGreedy, [0.02, 1.]],
    'epsilon-greedy-0.3': [solvers.EpsilonGreedy, [0.03, 1.]],
    'epsilon-greedy-0.4': [solvers.EpsilonGreedy, [0.04, 1.]],
}


def main():
    # prepare dump dir
    if not os.path.exists(DUMP_DIR):
        os.makedirs(DUMP_DIR)

    # compute
    for i in range(SEED_ITERS):
        print("Iter: {}/{}".format(i+1, SEED_ITERS))
        # random number generator
        rng = RandomState(int(time.time()))

        # randomly sample arm probabilities
        probas = [rng.rand() for _ in range(10)]

        # instantiate slot machine
        sm = machines.BernouilliSM(len(probas), probas, rng)

        # run algorithms
        for name, (algo, args) in ALGOS.items():
            print('\tBenchmarking {}...'.format(name))
            solver = algo(sm, *args)
            solver.solve(TIME_STEPS)
            solver.save(DUMP_DIR, i)
        time.sleep(1)

    # visualize
    vis = viz.Visualizer(DUMP_DIR)
    vis.plot(save=True)


if __name__ == '__main__':
    main()
