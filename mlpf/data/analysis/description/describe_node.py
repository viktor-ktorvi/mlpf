import hydra
import warnings

import pandas as pd

from pandas import DataFrame
from typing import Dict, List, Union

from mlpf.data.analysis.utils import ppc_list_extract_nodes, table_and_columns_from_config
from mlpf.data.analysis.description.describe import generate_description
from mlpf.data.loading.load_data import load_data, autodetect_load_ppc
from mlpf.enumerations.branch_table import BranchTableIds
from mlpf.enumerations.bus_table import BusTableIds
from mlpf.enumerations.gencost_table import GeneratorCostTableIds
from mlpf.enumerations.generator_table import GeneratorTableIds
from mlpf.enumerations.ppc_tables import get_table_ids, PPCTables


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


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg):
    """
    Load the dataset and describe a node in it. Use the hydra config default or overwrite the command line args.

    :param cfg: Hydra config. The config has the following fields:
    * columns: List[int] or null; the columns of the specified table
    * data_path: str; where to find the dataset
    * node_number: List[int]; which node to describe
    * table: str; ppc table string
    :return:
    """
    ppc_list = autodetect_load_ppc(cfg.data_path)

    table, columns = table_and_columns_from_config(cfg)

    pd.options.display.max_columns = None

    description = describe_nodes(ppc_list, table=table, node_numbers=cfg.node_numbers, columns=columns)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    print()
    print(f"Node # {cfg.node_numbers}")
    print(description)


if __name__ == "__main__":
    main()
