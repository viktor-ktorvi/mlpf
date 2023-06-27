from enum import IntEnum


class GeneratorCostTableIds(IntEnum):
    """
    Enumeration for the indices of the ppc gencost table.
    """
    model = 0
    startup_cost_USD = 1
    shutdown_cost_USD = 2
    parameter_number = 3
    coefficients_start = 4
