from typing import List, Union

from numpy import ndarray
from pandas import DataFrame

from mlpf.data.analysis.utils import generate_data_frame
from mlpf.enumerations.branch_table import BranchTableIds
from mlpf.enumerations.bus_table import BusTableIds
from mlpf.enumerations.gencost_table import GeneratorCostTableIds
from mlpf.enumerations.generator_table import GeneratorTableIds
from mlpf.enumerations.ppc_tables import PPCTables


def generate_description(dataset: ndarray,
                         table: PPCTables,
                         columns: List[Union[BusTableIds, GeneratorTableIds, BranchTableIds, GeneratorCostTableIds]] = None) -> DataFrame:
    """
    Generate the description DataFrame from the given dataset array, table and column.

    :param dataset: ndarray with the data to be described.
    :param table: PPCTables object.
    :param columns: List of table id enums specifying which columns to describe.
    :return: DataFrame object containing the description. To view the stats summary print the description DataFrame.
    """
    data_frame = generate_data_frame(dataset, table, columns)
    description = data_frame.describe()

    return description
