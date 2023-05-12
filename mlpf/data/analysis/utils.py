import numpy as np

from numpy import ndarray
from pandas import DataFrame
from typing import Dict, List, Union

from mlpf.enumerations.branch_table import BranchTableIds
from mlpf.enumerations.bus_table import BusTableIds
from mlpf.enumerations.bus_type import BusTypeIds
from mlpf.enumerations.gencost_table import GeneratorCostTableIds
from mlpf.enumerations.generator_table import GeneratorTableIds
from mlpf.enumerations.ppc_tables import PPCTables, get_table_ids


def ppc_extract_bus_type(ppc: Dict, table: PPCTables, bus_type: BusTypeIds) -> ndarray:
    """
    Extract the ndarray of the specified table from a ppc, but only with the buses of the specified bus_type.

    :param ppc: Pypower case file.
    :param table: PPCTables enum object.
    :param bus_type: BusTypeIds enum object.
    :return: ndarray of the specified table with only the buses of the specified type.
    """
    data_sample = ppc[table.value]

    bus_table_type_mask = bus_type.value == ppc["bus"][:, BusTableIds.bus_type]

    if table == PPCTables.Generator:
        valid_bus_numbers = ppc["bus"][bus_table_type_mask, BusTableIds.bus_number]
        gen_table_type_mask = np.isin(ppc["gen"][:, GeneratorTableIds.bus_number], valid_bus_numbers)
        data_sample = data_sample[gen_table_type_mask]

    elif table == PPCTables.Bus:
        data_sample = data_sample[bus_table_type_mask]

    else:
        # TODO support gencost
        raise ValueError(f"ppc_extract_bus_type currently supports only [{PPCTables.Bus}, {PPCTables.Generator}]. The given argument was {table}.")

    return data_sample


def ppc_list_extract_bus_type(ppc_list: List[Dict], table: PPCTables, bus_type: BusTypeIds = None) -> ndarray:
    """
    Extract an ndarray of the specified table for every ppc in a list and merge them to one ndarray, but only with the buses of the specified bus_type.

    :param ppc_list: List of pypower case files.
    :param table: PPCTables enum object.
    :param bus_type: BusTypeIds enum object.
    :return: ndarray of the specified table with only the buses of the specified type.
    """
    data_list = []
    for ppc in ppc_list:

        if bus_type is None:
            data_sample = ppc[table.value]
        else:
            data_sample = ppc_extract_bus_type(ppc, table, bus_type)

        data_list.append(data_sample)

    return np.vstack(data_list)


def ppc_extract_node(ppc: Dict, table: PPCTables, node_number: int) -> ndarray:
    """
    Extract the ndarray of the specified table from a ppc for the specified node.

    :param ppc: Pypower case file.
    :param table: PPCTables enum object.
    :param node_number: The bus table number of the node of interest.
    :return: ndarray of the specified table with only the specified node.
    """
    data_sample = ppc[table.value]

    if table == PPCTables.Generator:
        data_sample = data_sample[ppc["gen"][:, GeneratorTableIds.bus_number] == node_number]

    elif table == PPCTables.Bus:
        data_sample = data_sample[node_number]

    else:
        # TODO support gencost
        raise ValueError(f"ppc_extract_bus_type currently supports only [{PPCTables.Bus}, {PPCTables.Generator}]. The given argument was {table}.")

    return data_sample


def ppc_list_extract_node(ppc_list: List[Dict], table: PPCTables, node_number: int) -> ndarray:
    """
    Extract an ndarray of the specified table for every ppc in a list, for the specified node.

    :param ppc_list: List of pypower case files.
    :param table: PPCTables enum object.
    :param node_number: The bus table number of the node of interest.
    :return: ndarray of the specified table with only the specified node.
    """
    data_list = []
    for ppc in ppc_list:
        data_sample = ppc_extract_node(ppc, table=table, node_number=node_number)
        data_list.append(data_sample)

    return np.vstack(data_list)


def generate_data_frame(dataset: ndarray,
                        table: PPCTables,
                        columns: List[Union[BusTableIds, GeneratorTableIds, BranchTableIds, GeneratorCostTableIds]] = None) -> DataFrame:
    """
    Generate a DataFrame from the given dataset array, table and column.

    :param dataset: ndarray with the data to be collected.
    :param table: PPCTables object.
    :param columns: List of table id enums specifying which columns to collect.
    :return: DataFrame object containing the columns of the dataset.
    """
    table_ids_enum = get_table_ids(table)

    # extract column names; set index as name if name isn't defined
    table_column_enum_values = set(item.value for item in table_ids_enum)
    column_names = [table_ids_enum(i).name if i in table_column_enum_values else str(i) for i in range(dataset.shape[1])]

    if columns is not None:
        # check if columns match table
        for column in columns:
            if column not in table_ids_enum:
                raise ValueError(f"Column {column} doesn't exist in table {table}")

        # select column ids
        column_ids = [column.value for column in columns]
        column_names = [column_names[i] for i in column_ids]
    else:
        column_ids = [i for i in range(dataset.shape[1])]

    return DataFrame(data=dataset[:, column_ids], columns=column_names)


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
