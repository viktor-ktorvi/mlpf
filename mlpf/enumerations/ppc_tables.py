from enum import Enum
from typing import Type, Any

from mlpf.enumerations.branch_table import BranchTableIds
from mlpf.enumerations.bus_table import BusTableIds
from mlpf.enumerations.gencost_table import GeneratorCostTableIds
from mlpf.enumerations.generator_table import GeneratorTableIds


class PPCTables(Enum):
    """
    Enumeration for all the PPC fields/tables.
    """
    BaseMVA = "baseMVA"
    Branch = "branch"
    Bus = "bus"
    Generator = "gen"
    GeneratorCost = "gencost"


def get_table_ids(table: PPCTables) -> Type[Any]:
    """
    Get the table ids enum corresponding to the given table.

    :param table: PPCTables object representing a table in the pypower case format.
    :return: Table ids enum
    """
    # get the corresponding table ids
    if table is PPCTables.Bus:
        table_ids_enum = BusTableIds

    elif table is PPCTables.Branch:
        table_ids_enum = BranchTableIds

    elif table is PPCTables.Generator:
        table_ids_enum = GeneratorTableIds

    elif table is PPCTables.GeneratorCost:
        table_ids_enum = GeneratorCostTableIds

    else:
        raise NotImplementedError(f"Table argument {table} is not supported.")

    return table_ids_enum
