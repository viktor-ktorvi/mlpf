from enum import IntEnum


class BusTableIds(IntEnum):
    """
    Enumeration for the indices of the ppc bus table.
    """
    bus_number = 0
    bus_type = 1
    active_power_MW = 2
    reactive_power_MVAr = 3
    shunt_conductance = 4
    shunt_susceptance = 5
    area_number = 6
    voltage_magnitude_pu = 7
    voltage_angle_deg = 8
    base_kV = 9
    loss_zone = 10
    voltage_max_pu = 11
    voltage_min_pu = 12
