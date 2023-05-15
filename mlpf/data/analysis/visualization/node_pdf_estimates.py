import hydra

from typing import Dict, List, Union
from matplotlib import pyplot as plt

from mlpf.data.analysis.utils import generate_data_frame, ppc_list_extract_node, table_and_columns_from_config, create_subplots_grid
from mlpf.data.analysis.visualization.visualize import visualize_pdf_data_frame
from mlpf.data.loading.load_data import load_data
from mlpf.enumerations.branch_table import BranchTableIds
from mlpf.enumerations.bus_table import BusTableIds
from mlpf.enumerations.gencost_table import GeneratorCostTableIds
from mlpf.enumerations.generator_table import GeneratorTableIds
from mlpf.enumerations.ppc_tables import get_table_ids, PPCTables


# TODO test for crashes on all grids
def visualize_node_pdfs(ppc_list: List[Dict],
                        table: PPCTables,
                        node_number: int = 0,
                        columns: List[Union[BusTableIds, GeneratorTableIds, BranchTableIds, GeneratorCostTableIds]] = None,
                        kernel: str = "tophat", bandwidth_coeff: float = 0.05,
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


def visualize_node_histograms(ppc_list: List[Dict],
                              table: PPCTables,
                              node_number: int = 0,
                              columns: List[Union[BusTableIds, GeneratorTableIds, BranchTableIds, GeneratorCostTableIds]] = None):
    """
    Estimate and plot the histogram of each specified column for the specified node and table in the ppc list.

    :param ppc_list: List of pypower case files.
    :param table: PPCTables object specifying which table to describe.
    :param node_number: The bus number in the bus table of the node to describe.
    :param columns: List of table id enums specifying which columns to describe.
    be one supported by this function.
    :return:
    """

    dataset = ppc_list_extract_node(ppc_list, table, node_number=node_number)
    data_frame = generate_data_frame(dataset, table, columns)

    data_frame.hist()


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg):
    data_list = load_data(cfg.data_path)

    table, columns = table_and_columns_from_config(cfg)

    fig, axes = create_subplots_grid(len(columns))

    fig.tight_layout()
    visualize_node_pdfs(data_list, table, node_number=cfg.node_number, columns=columns, kernel=cfg.visualization.kernel, bandwidth_coeff=cfg.visualization.bandwidth_coeff,
                        axes=axes)

    for ax in axes.flatten():
        ax.set_ylim(bottom=0)

    visualize_node_histograms(data_list, table, node_number=cfg.node_number, columns=columns)

    plt.show()


if __name__ == "__main__":
    main()
