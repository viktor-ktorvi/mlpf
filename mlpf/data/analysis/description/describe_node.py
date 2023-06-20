from typing import Dict, List, Union

from pandas import DataFrame

from mlpf.data.analysis.description.describe import generate_description
from mlpf.data.analysis.utils import ppc_list_extract_nodes
from mlpf.enumerations.branch_table import BranchTableIds
from mlpf.enumerations.bus_table import BusTableIds
from mlpf.enumerations.gencost_table import GeneratorCostTableIds
from mlpf.enumerations.generator_table import GeneratorTableIds
from mlpf.enumerations.ppc_tables import PPCTables


def describe_nodes(ppc_list: List[Dict],
                   table: PPCTables,
                   node_numbers: List[int],
                   columns: List[Union[BusTableIds, GeneratorTableIds, BranchTableIds, GeneratorCostTableIds]] = None) -> DataFrame:
    """
    Return a description DataFrame similar to pandas' _describe_ function.

    :param ppc_list: List of pypower case files.
    :param table: PPCTables object specifying which table to describe.
    :param node_numbers: The bus number in the bus table of the node to describe.
    :param columns: List of table id enums specifying which columns to describe.
    :return: DataFrame object containing the description. To view the stats summary print the description DataFrame.
    """

    dataset = ppc_list_extract_nodes(ppc_list, table, node_numbers)
    return generate_description(dataset, table, columns)
