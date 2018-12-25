"""Convenience class for visualizing MAB algorithm performance.
"""

import glob
import os

import numpy as np
from ipdb import set_trace as debug
from matplotlib import pyplot as plt


class Visualizer(object):
    """A visualization API for MAB solvers.
    """
    def __init__(self, dirname):
        self.dirname = dirname

        # grab all .npy filenames
        filenames = list(map(os.path.basename, glob.glob(dirname + "*.npy")))

        # get list of different algorithms from prefix
        prefixes = list(map(lambda x: x.split('_')[0], filenames))
        self.prefixes = list(set(prefixes))

        # group filenames by algorithm prefix
        metrics = ['counter', 'rewards', 'regrets']
        self.filenames = {
            key: {subkey: [] for subkey in metrics} for key in self.prefixes
        }
        for name in filenames:
            prefix, suffix = name.split('.')[0].split('_')
            self.filenames[prefix][suffix].append(name)

    def plot_rewards(self):
        """Plots raw and cumulative rewards for all algorithms."""
