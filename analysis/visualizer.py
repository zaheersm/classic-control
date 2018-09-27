import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class RunLines(object):
    """
    Draws lines with multiple runs along with their error bars
    """
    def __init__(self, path_formatters, num_runs, num_datapoints, labels, parser_func=None, save_path=None, xlabel=None, ylabel=None):
        """
        :param path_formatters: list of generic data paths for each line
                                in which run number can be substituted
        :param num_runs: list of number of runs for each algorithm
        :param num_datapoints: number of datapoints to be expected for each line
        :param parser_func: function to be used for parsing the data file
        :param save_path: save_path to store the plot
        """
        assert len(path_formatters) > 0
        assert len(path_formatters) == len(num_runs)
        assert len(path_formatters) == len(num_datapoints)
        assert len(path_formatters) == len(labels)
        self.path_formatters = path_formatters
        self.num_runs = num_runs
        self.num_datapoints = num_datapoints
        self.parser_func = parser_func
        self.save_path = save_path
        self.labels = labels
        self.xlabel = xlabel
        self.ylabel = ylabel

    def draw(self):
        sns.set(style="darkgrid")
        for idx, pf in enumerate(self.path_formatters):
            nr = self.num_runs[idx]
            nd = self.num_datapoints[idx]
            label = self.labels[idx]
            lines = np.zeros((nr, nd))
            for run in range(nr):
                path = pf.format(run)
                line = self.parser_func(path, nd)
                lines[run, :] = line
            mean = np.nanmean(lines, axis=0)
            std = np.nanstd(lines, axis=0)
            plt.fill_between(range(nd), mean - std, mean + std,
                             alpha=0.1)
            plt.plot(range(nd), mean, label=label)
        # labels = map(lambda x: str(int((x * 10000 / 1000))) + 'K', range(0, 5000, 100))
        # plt.xticks(range(0, 1001, 100), labels)
        plt.legend(loc="best", frameon=False)
        if self.xlabel is not None:
            plt.xlabel(self.xlabel)
        if self.ylabel is not None:
            plt.ylabel(self.ylabel)
        plt.savefig(self.save_path)






