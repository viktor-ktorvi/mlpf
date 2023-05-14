import hydra
import matplotlib.pylab

import numpy as np
from numpy import ndarray
from pandas import DataFrame

from sklearn.neighbors import KernelDensity
from typing import Dict, List, Union
from matplotlib import pyplot as plt

from mlpf.data.analysis.utils import generate_data_frame, ppc_list_extract_node
from mlpf.data.loading.load_data import load_data
from mlpf.enumerations.branch_table import BranchTableIds
from mlpf.enumerations.bus_table import BusTableIds
from mlpf.enumerations.gencost_table import GeneratorCostTableIds
from mlpf.enumerations.generator_table import GeneratorTableIds
from mlpf.enumerations.ppc_tables import get_table_ids, PPCTables


# TODO test for crashes on all grids
def visualize_node_pdf(ppc_list: List[Dict], table: PPCTables,
                       node_number: int = 0,
                       columns: List[Union[BusTableIds, GeneratorTableIds, BranchTableIds, GeneratorCostTableIds]] = None, kernel: str = "tophat", bandwidth_coeff: float = 0.05,
                       axes=None):
    """
    Estimate and plot the probability density function of each specified column for the specified node and table in the ppc list. If no axes list is provided
    new figures will be created.

    :param ppc_list: List of pypower case files.
    :param table: PPCTables object specifying which table to describe.
    :param node_number: The bus number in the bus table of the node to describe.
    :param columns: List of table id enums specifying which columns to describe.
    :param kernel: Name of kernel method to use. The function calls sklearn.neighbors.KernelDensity so it must
    be one supported by this function.
    :param bandwidth_coeff: A parameter that effects the resolution of the pdf estimate. Too small of a value
    will overfit to the given samples. To large of a value will not capture enough detail.
    :param axes: A numpy array of Matplotlib.PyPlot.Axes objects onto which to plot the estimates. If None,
    new figures will be created for every column.
    :return:
    """

    dataset = ppc_list_extract_node(ppc_list, table, node_number=node_number)
    data_frame = generate_data_frame(dataset, table, columns)

    visualize_pdf_data_frame(data_frame, kernel, bandwidth_coeff, axes)


def visualize_pdf_data_frame(data_frame: DataFrame,
                             kernel: str = "tophat",
                             bandwidth_coeff: float = 0.05,
                             axes: ndarray[matplotlib.pylab.Axes] = None):
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


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg):
    data_list = load_data(cfg.data_path)

    table = PPCTables(cfg.table)

    if cfg.columns is None:
        columns = None
    else:
        table_ids_enum = get_table_ids(table)
        columns = [table_ids_enum(i) for i in cfg.columns]

    if len(columns) % 2 == 0:
        fig, axes = plt.subplots(2, len(columns) // 2)
    else:
        fig, axes = plt.subplots(2, len(columns) // 2 + 1)
        fig.delaxes(axes.flatten()[-1])

    fig.tight_layout()
    visualize_node_pdf(data_list, table, node_number=cfg.node_number, columns=columns, kernel=cfg.visualization.kernel, bandwidth_coeff=cfg.visualization.bandwidth_coeff,
                       axes=axes)

    plt.show()


if __name__ == "__main__":
    main()
