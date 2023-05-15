import numpy as np

from matplotlib.axes import Axes
from numpy import ndarray
from pandas import DataFrame
from sklearn.neighbors import KernelDensity

from matplotlib import pyplot as plt


def visualize_pdf_data_frame(data_frame: DataFrame,
                             kernel: str = "tophat",
                             bandwidth_coeff: float = 0.05,
                             axes: ndarray[Axes] = None,
                             max_samples: int = 2500):
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
    :param max_samples: Since estimating the pdf could take a lot of time for a large number of samples, _max_samples_ provides a
    limit on the number of samples to take into consideration. If the number of samples is larger than _max_samples_ then approximately
    _max_samples_ samples are randomly sampled.
    :return:
    """
    for i, column in enumerate(data_frame):

        if axes is None:
            fig, ax = plt.subplots()
        else:
            ax = axes.flatten()[i]

        x_values = data_frame[column]

        plot_pdf(x_values, ax, max_samples, kernel, bandwidth_coeff)

        ax.set_title(column)
        ax.set_xlabel(column)


def visualize_histogram_data_frame(data_frame: DataFrame,
                                   bins: int = 10,
                                   axes: ndarray[Axes] = None):
    """
    Estimate and plot the probability density function of each column of the given dataframe. If no axes list is provided
    new figures will be created.

    :param data_frame: DataFrame to visualize.
    :param bins: Number of bins to consider.
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

        ax.hist(x_values, bins=bins)
        ax.set_title(column)
        ax.set_xlabel(column)


def plot_pdf(x_values: ndarray,
             ax: Axes,
             max_samples: int = 2500,
             kernel: str = "tophat",
             bandwidth_coeff: float = 0.05):
    """
    Estimate and plot the probability density function of the given array.

    :param x_values: An ndarray to estimate the pdf of.
    :param kernel: Name of kernel method to use. The function calls sklearn.neighbors.KernelDensity so it must
    be one supported by this function.
    :param bandwidth_coeff: A parameter that effects the resolution of the pdf estimate. Too small of a value
    will overfit to the given samples. To large of a value will not capture enough detail.
    :param ax: Matplotlib.PyPlot.Axes object onto which to plot the estimate.
    :param max_samples: Since estimating the pdf could take a lot of time for a large number of samples, _max_samples_ provides a
    limit on the number of samples to take into consideration. If the number of samples is larger than _max_samples_ then approximately
    _max_samples_ samples are randomly sampled.
    :return:
    """

    if len(x_values) > max_samples:
        # shuffle and sample a fraction
        x_values = x_values.sample(frac=max_samples / len(x_values))

    x_values = x_values.sort_values().to_numpy().reshape(-1, 1)

    total_width = np.max(x_values) - np.min(x_values)

    if total_width > 0.0:
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth_coeff * total_width).fit(x_values)
        pdf_estimate = np.exp(kde.score_samples(x_values))

        ax.plot(x_values, pdf_estimate)

    else:
        # if the values are all the same value it makes sense to just do a histogram/bar plot
        ax.hist(x_values)
