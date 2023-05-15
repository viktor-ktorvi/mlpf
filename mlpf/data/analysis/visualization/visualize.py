import numpy as np

from matplotlib.pylab import Axes
from numpy import ndarray
from pandas import DataFrame
from sklearn.neighbors import KernelDensity

from matplotlib import pyplot as plt


def visualize_pdf_data_frame(data_frame: DataFrame,
                             kernel: str = "tophat",
                             bandwidth_coeff: float = 0.05,
                             axes: ndarray[Axes] = None):
    """
    Estimate and plot the probability density function of each column of the given dataframe. If no axes list is provided
    new figures will be created.

    :param data_frame: DataFrame to visualize.
    :param kernel: Name of kernel method to use. The function calls sklearn.neighbors.KernelDensity so it must
    be one supported by this function.
    :param bandwidth_coeff: A parameter that effects the resolution of the pdf estimate. Too small of a value
    will overfit to the given samples. To large of a value will not capture enough detail.
    :param axes: A numpy array of Matplotlib.PyPlot.Axes objects onto which to plot the estimates. If None,
    new figures will be created for every column.
    :return:
    """
    for i, column in enumerate(data_frame):

        if axes is None:
            fig, ax = plt.subplots()
        else:
            ax = axes.flatten()[i]

        x_values = data_frame[column]
        x_values = x_values.sort_values().to_numpy().reshape(-1, 1)

        total_width = np.max(x_values) - np.min(x_values)

        if total_width > 0.0:
            kde = KernelDensity(kernel=kernel, bandwidth=bandwidth_coeff * total_width).fit(x_values)
            pdf_estimate = np.exp(kde.score_samples(x_values))

            ax.plot(x_values, pdf_estimate)

        else:
            ax.hist(x_values)

        ax.set_title(column)
        ax.set_xlabel(column)
