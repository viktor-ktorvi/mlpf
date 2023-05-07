from enum import IntEnum


class GenTableIds(IntEnum):
    """
    Enumeration for the indices of the ppc bus table.
    """
    bus_number = 0
    active_power = 1  # MW
    reactive_power = 2  # MVar
    max_reactive_power = 3  # Mvar
    min_reactive_power = 4  # Mvar
    max_active_power = 8  # MW
    min_active_power = 9  # MW
