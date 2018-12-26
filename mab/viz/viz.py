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
            prefix, _, suffix = name.split('.')[0].split('_')
            self.filenames[prefix][suffix].append(name)

    def plot_rewards(self):
        """Plots raw rewards for each algorithm."""
        num_plots = len(self.prefixes)
        colors = ['r', 'b', 'g', 'o']
        labels = []
        fig, axes = plt.subplots(1, num_plots, figsize=(15, 15))
        for i, key in enumerate(self.filenames.keys()):
            if key == "ts":
                label = "thompson-sampling"
            elif key == "eg":
                label = "epsilon-greedy"
            else:
                raise ValueError()
            filenames = self.filenames[key]['rewards']
            rewards = []
            for name in filenames:
                rewards.append(np.load(os.path.join(self.dirname, name)))
            rewards = np.mean(rewards, axis=0)
            axes[i].plot(rewards, label=label, color=colors[i])
        plt.show()

    def plot_cumrewards(self):
        """Plots cumulative rewards for each algorithm."""
        colors = ['r', 'b', 'g', 'o']
        labels = []
        fig, ax = plt.subplots(figsize=(15, 15))
        for i, key in enumerate(self.filenames.keys()):
            if key == "ts":
                label = "thompson-sampling"
            elif key == "eg":
                label = "epsilon-greedy"
            else:
                raise ValueError()
            filenames = self.filenames[key]['rewards']
            rewards = []
            for name in filenames:
                rewards.append(
                    np.cumsum(
                        np.load(os.path.join(self.dirname, name))
                    )
                )
            rewards = np.mean(rewards, axis=0)
            ax.plot(rewards, label=label, color=colors[i])
        plt.legend()
        plt.show()
