from typing import Dict, List, Union

from matplotlib.axes import Axes
from numpy import ndarray

from mlpf.data.analysis.utils import generate_data_frame, ppc_list_extract_nodes
from mlpf.data.analysis.visualization.visualize import visualize_pdf_data_frame, visualize_histogram_data_frame
from mlpf.enumerations.branch_table import BranchTableIds
from mlpf.enumerations.bus_table import BusTableIds
from mlpf.enumerations.gencost_table import GeneratorCostTableIds
from mlpf.enumerations.generator_table import GeneratorTableIds
from mlpf.enumerations.ppc_tables import PPCTables


# TODO test for crashes on all grids
def visualize_node_pdfs(ppc_list: List[Dict],
                        table: PPCTables,
                        node_numbers: List[int],
                        columns: List[Union[BusTableIds, GeneratorTableIds, BranchTableIds, GeneratorCostTableIds]] = None,
                        kernel: str = "tophat",
                        bandwidth_coeff: float = 0.05,
                        axes: ndarray[Axes] = None):
    """
    Estimate and plot the probability density function of each specified column for the specified node and table in the ppc list. If no axes list is provided
    new figures will be created.

    :param ppc_list: List of pypower case files.
    :param table: PPCTables object specifying which table to describe.
    :param node_numbers: The bus number in the bus table of the node to describe.
    :param columns: List of table id enums specifying which columns to describe.
    :param kernel: Name of kernel method to use. The function calls sklearn.neighbors.KernelDensity so it must
    be one supported by this function.
    :param bandwidth_coeff: A parameter that effects the resolution of the pdf estimate. Too small of a value
    will overfit to the given samples. To large of a value will not capture enough detail.
    :param axes: A numpy array of Matplotlib.PyPlot.Axes objects onto which to plot the estimates. If None,
    new figures will be created for every column.
    :return:
    """

    dataset = ppc_list_extract_nodes(ppc_list, table, node_numbers=node_numbers)
    data_frame = generate_data_frame(dataset, table, columns)

    visualize_pdf_data_frame(data_frame, kernel, bandwidth_coeff, axes)


def visualize_node_histograms(ppc_list: List[Dict],
                              table: PPCTables,
                              node_numbers: List[int],
                              columns: List[Union[BusTableIds, GeneratorTableIds, BranchTableIds, GeneratorCostTableIds]] = None,
                              bins: int = 10,
                              axes: ndarray[Axes] = None):
    """
    Estimate and plot the histogram of each specified column for the specified node and table in the ppc list.

    :param ppc_list: List of pypower case files.
    :param table: PPCTables object specifying which table to describe.
    :param node_numbers: The bus number in the bus table of the node to describe.
    :param columns: List of table id enums specifying which columns to describe.
    be one supported by this function.
    :param bins: How many bins to work with.
    :param axes: Ndarray of Axes
    :return:
    """

    dataset = ppc_list_extract_nodes(ppc_list, table, node_numbers=node_numbers)
    data_frame = generate_data_frame(dataset, table, columns)

    visualize_histogram_data_frame(data_frame, bins, axes)
