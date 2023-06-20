from typing import Dict, List, Union, Tuple, Any

import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame

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


def ppc_extract_nodes(ppc: Dict, table: PPCTables, node_numbers: List[int]) -> ndarray:
    """
    Extract the ndarray of the specified table from a ppc for the specified nodes.

    :param ppc: Pypower case file.
    :param table: PPCTables enum object.
    :param node_numbers: The bus table numbers of the nodes of interest.
    :return: ndarray of the specified table with only the specified node.
    """
    data_sample = ppc[table.value]

    if table == PPCTables.Generator:
        data_sample = data_sample[np.isin(ppc["gen"][:, GeneratorTableIds.bus_number], node_numbers)]

    elif table == PPCTables.Bus:
        data_sample = data_sample[node_numbers]

    else:
        # TODO support gencost
        raise ValueError(f"ppc_extract_bus_type currently supports only [{PPCTables.Bus}, {PPCTables.Generator}]. The given argument was {table}.")

    return data_sample


def ppc_list_extract_nodes(ppc_list: List[Dict], table: PPCTables, node_numbers: List[int]) -> ndarray:
    """
    Extract an ndarray of the specified table for every ppc in a list, for the specified nodes.

    :param ppc_list: List of pypower case files.
    :param table: PPCTables enum object.
    :param node_numbers: The bus table numbers of the nodes of interest.
    :return: ndarray of the specified table with only the specified node.
    """
    data_list = []
    for ppc in ppc_list:
        data_sample = ppc_extract_nodes(ppc, table=table, node_numbers=node_numbers)
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


def table_and_columns_from_config(cfg) -> Tuple[PPCTables, Union[List[Any], None]]:
    """
    Extract the table and column objects using the given config.

    :param cfg: Hydra config.
    :return: PPCTables
    """
    table = PPCTables(cfg.table)

    if cfg.columns is None:
        columns = None
    else:
        table_ids_enum = get_table_ids(table)
        columns = [table_ids_enum(i) for i in cfg.columns]

    return table, columns


def create_subplots_grid(num_axes: int):
    """
    Create a (2, len(columns)) subplot.

    :param num_axes: number of axes in total
    :return: figure and axes array
    """

    assert num_axes > 0

    # TODO this could be generalized to numbers that aren't 2
    if num_axes % 2 == 0:
        fig, axes = plt.subplots(2, num_axes // 2)
    else:
        fig, axes = plt.subplots(2, num_axes // 2 + 1)
        fig.delaxes(axes.flatten()[-1])

    return fig, axes
