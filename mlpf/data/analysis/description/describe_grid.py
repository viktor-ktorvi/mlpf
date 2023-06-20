from typing import Dict, List, Union

from pandas import DataFrame

from mlpf.data.analysis.description.describe import generate_description
from mlpf.data.analysis.utils import ppc_list_extract_bus_type
from mlpf.enumerations.branch_table import BranchTableIds
from mlpf.enumerations.bus_table import BusTableIds
from mlpf.enumerations.bus_type import BusTypeIds
from mlpf.enumerations.gencost_table import GeneratorCostTableIds
from mlpf.enumerations.generator_table import GeneratorTableIds
from mlpf.enumerations.ppc_tables import PPCTables


def describe_grid(ppc_list: List[Dict],
                  table: PPCTables,
                  bus_type: BusTypeIds = None,
                  columns: List[Union[BusTableIds, GeneratorTableIds, BranchTableIds, GeneratorCostTableIds]] = None) -> DataFrame:
    """
    Return a description DataFrame similar to pandas' _describe_ function.

    :param ppc_list: List of pypower case files
    :param table: PPCTables object specifying which table to describe.
    :param bus_type: BusTypeIds object specifying which bus types to describe.
    :param columns: List of table id enums specifying which columns to describe.
    :return: DataFrame object containing the description. To view the stats summary print the description DataFrame.
    """

    dataset = ppc_list_extract_bus_type(ppc_list, table, bus_type)

    return generate_description(dataset, table, columns)
