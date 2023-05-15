import hydra
import warnings

import pandas as pd

from pandas import DataFrame
from typing import Dict, List, Union

from mlpf.data.analysis.utils import generate_description, ppc_list_extract_node
from mlpf.data.loading.load_data import load_data
from mlpf.enumerations.branch_table import BranchTableIds
from mlpf.enumerations.bus_table import BusTableIds
from mlpf.enumerations.gencost_table import GeneratorCostTableIds
from mlpf.enumerations.generator_table import GeneratorTableIds
from mlpf.enumerations.ppc_tables import get_table_ids, PPCTables


def describe_node(ppc_list: List[Dict],
                  table: PPCTables,
                  node_number: int,
                  columns: List[Union[BusTableIds, GeneratorTableIds, BranchTableIds, GeneratorCostTableIds]] = None) -> DataFrame:
    """
    Return a description DataFrame similar to pandas' _describe_ function.

    :param ppc_list: List of pypower case files.
    :param table: PPCTables object specifying which table to describe.
    :param node_number: The bus number in the bus table of the node to describe.
    :param columns: List of table id enums specifying which columns to describe.
    :return: DataFrame object containing the description. To view the stats summary print the description DataFrame.
    """

    dataset = ppc_list_extract_node(ppc_list, table, node_number)
    return generate_description(dataset, table, columns)


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg):
    """
    Load the dataset and describe a node in it. Use the hydra config default or overwrite the command line args.

    :param cfg: Hydra config. The config has the following fields:
    * columns: List[int] or null; the columns of the specified table
    * data_path: str; where to find the dataset
    * node_number: int; which node to describe
    * table: str; ppc table string
    :return:
    """
    data_list = load_data(cfg.data_path)

    table = PPCTables(cfg.table)

    if cfg.columns is None:
        columns = None
    else:
        table_ids_enum = get_table_ids(table)
        columns = [table_ids_enum(i) for i in cfg.columns]

    warnings.filterwarnings('ignore')  # ComplexWarning error
    pd.options.display.max_columns = None

    description = describe_node(data_list, table=table, node_number=cfg.node_number, columns=columns)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    print()
    print(f"Node #{cfg.node_number}")
    print(description)


if __name__ == "__main__":
    main()