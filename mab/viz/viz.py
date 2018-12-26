"""Convenience class for visualizing MAB algorithm performance.
"""

import glob
import os

import numpy as np
from matplotlib import pyplot as plt


class Visualizer(object):
    """A visualization API for MAB solvers.
    """
    def __init__(self, dirname):
        self.dirname = dirname

        # grab all .npy filenames
        filenames = list(map(os.path.basename, glob.glob(dirname + "*.npy")))

        metrics = ['counter', 'rewards', 'regrets']

        # get list of different algorithms from prefix
        prefixes = ['-'.join(x.split('-')[:-2]) for x in filenames]
        self.prefixes = list(set(prefixes))

        # group filenames by algorithm prefix
        self.filenames = {
            key: {subkey: [] for subkey in metrics} for key in self.prefixes
        }
        for name in filenames:
            splits = name.split('-')
            prefix = '-'.join(splits[:-2])
            suffix = splits[-1].split('.')[0]
            self.filenames[prefix][suffix].append(name)

    def plot(self, save):
        """Plots cumulative reward and counts for each algorithm."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, key in enumerate(self.filenames.keys()):
            # compute average cumulative reward
            filenames = self.filenames[key]['rewards']
            rewards = []
            for name in filenames:
                rewards.append(np.cumsum(np.load(os.path.join(self.dirname, name))))
            rewards = np.mean(rewards, axis=0)
            # compute average cumulative regret
            filenames = self.filenames[key]['regrets']
            regrets = []
            for name in filenames:
                regrets.append(np.cumsum(np.load(os.path.join(self.dirname, name))))
            regrets = np.mean(regrets, axis=0)
            # compute average arm counts
            filenames = self.filenames[key]['counter']
            counts = []
            for name in filenames:
                counts.append(np.load(os.path.join(self.dirname, name)))
            counts = np.mean(counts, axis=0)
            # plot
            axes[0].plot(rewards, label=key)
            axes[1].plot(regrets, label=key)
            axes[2].plot(np.arange(1, 11), counts, label=key, alpha=0.5)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Cumulative Reward')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Cumulative Regret')
        axes[2].set_xlabel('Arm')
        axes[2].set_ylabel('Pull Frequency')
        axes[2].set_xticks(np.arange(1, 11, 1.0))
        for ax in axes.flatten():
            ax.grid('k', ls='--', alpha=0.3)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center')
        plt.subplots_adjust(wspace=0.2)
        if save:
            plt.savefig("./assets/performance.png", format="png", dpi=300)
        plt.show()
