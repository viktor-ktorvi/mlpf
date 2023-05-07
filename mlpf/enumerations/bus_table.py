from enum import IntEnum


class BusTableIds(IntEnum):
    """
    Enumeration for the indices of the ppc bus table.
    """
    bus_number = 0
    bus_type = 1
    active_power = 2  # MW
    reactive_power = 3  # MVar
    voltage_magnitude = 7  # p.u.
    voltage_angle = 8  # degree
    base_kV = 9  # kV
    voltage_max = 11  # p.u.
    voltage_min = 12  # p.u.
