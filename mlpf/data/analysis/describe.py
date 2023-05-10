import hydra
import warnings

import numpy as np
import pandas as pd

from enum import IntEnum
from pandas import DataFrame
from typing import Dict, List

from mlpf.data.loading.load_data import load_data
from mlpf.enumerations.ppc_tables import get_table_ids, PPCTables


#  TODO test inputs
def describe_ppc(ppc_list: List[Dict], table: PPCTables, columns: List[IntEnum] = None, node_id: int = None) -> DataFrame:
    """
    Describe the pypower case format dataset using pandas' describe function. Choose which table in the PPC to describe and optionally which columns.
    If columns is None(default value) describe the entire table. Return the description dataframe.

    :param ppc_list: List of pypower case files.
    :param table: PPCTables enum object to choose the PPC table.
    :param columns: List of enums of table ids corresponding to the given table. None by default.
    :param node_id: Index of the node for which to describe the values. If None then describe for all nodes. Default is None.
    :return: Description DataFrame.
    """

    # get the corresponding table ids
    table_ids_enum = get_table_ids(table)

    # extract the data
    if node_id is None:
        numpy_array_list = [ppc_list[i][table.value] for i in range(len(ppc_list))]
    else:
        numpy_array_list = [ppc_list[i][table.value][node_id, :] for i in range(len(ppc_list))]
    data = np.vstack(numpy_array_list)

    # extract column names; set index as name if name isn't defined
    table_column_enum_values = set(item.value for item in table_ids_enum)
    column_names = [table_ids_enum(i).name if i in table_column_enum_values else str(i) for i in range(data.shape[1])]

    if columns is not None:
        # check if columns match table
        for column in columns:
            if column not in table_ids_enum:
                raise ValueError(f"Column {column} doesn't exist in table {table}")

        # select column ids
        column_ids = [column.value for column in columns]
        column_names = [column_names[i] for i in column_ids]
    else:
        column_ids = [i for i in range(data.shape[1])]

    data_frame = DataFrame(data=data[:, column_ids], columns=column_names)
    description = data_frame.describe()

    return description


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg):
    data_list = load_data(cfg.data_path)

    table = PPCTables(cfg.table)
    if cfg.columns is None:
        columns = None
    else:
        table_ids_enum = get_table_ids(table)
        columns = [table_ids_enum(i) for i in cfg.columns]

    warnings.filterwarnings('ignore')  # ComplexWarning error
    pd.options.display.max_columns = None

    description = describe_ppc(data_list, table=table, columns=columns, node_id=cfg.node)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(description)


if __name__ == "__main__":
    main()
