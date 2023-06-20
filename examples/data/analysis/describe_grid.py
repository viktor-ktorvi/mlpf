import hydra
import pandas as pd

from mlpf.data.analysis.description.describe_grid import describe_grid
from mlpf.data.analysis.utils import table_and_columns_from_config
from mlpf.data.loading.load_data import autodetect_load_ppc
from mlpf.enumerations.bus_type import BusTypeIds


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg):
    """
    Load the dataset and describe it. Use the hydra config default or overwrite the command line args.

    :param cfg: Hydra config. The config has the following fields:
    * bus_type: int or null; the values follow the ppc convention
    * columns: List[int] or null; the columns of the specified table
    * data_path: str; where to find the dataset
    * table: str; ppc table string
    :return:
    """
    ppc_list = autodetect_load_ppc(cfg.data_path)

    bus_type = BusTypeIds(cfg.bus_type) if cfg.bus_type is not None else None
    table, columns = table_and_columns_from_config(cfg)

    pd.options.display.max_columns = None

    description = describe_grid(ppc_list, table=table, bus_type=bus_type, columns=columns)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    print()
    print("Bus type:", bus_type.name if bus_type is not None else "all")
    print(description)


if __name__ == "__main__":
    main()
