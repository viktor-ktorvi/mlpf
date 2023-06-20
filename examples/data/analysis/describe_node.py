import hydra
import pandas as pd

from mlpf.data.analysis.description.describe_node import describe_nodes
from mlpf.data.analysis.utils import table_and_columns_from_config
from mlpf.data.loading.load_data import autodetect_load_ppc


@hydra.main(version_base=None, config_path="configs", config_name="default")
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
