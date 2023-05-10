from enum import IntEnum


class GeneratorTableIds(IntEnum):
    """
    Enumeration for the indices of the ppc bus table.
    """
    bus_number = 0
    active_power_MW = 1
    reactive_power_MVAr = 2
    max_reactive_power_MVAr = 3
    min_reactive_power_MVAr = 4
    voltage_magnitude_setpoint_pu = 5
    total_MVA_base = 6
    status = 7
    max_active_power_MW = 8
    min_active_power_MW = 9
    Pc1_MW = 10
    Pc2_MW = 11
    Qc1min = 12
    Qc1max = 13
    Qc2min = 14
    Qc2max = 15
    ramp_rate_load_following_MW_per_min = 16
    ramp_rate_10_min_reserves_MW = 17
    ramp_rate_30_min_reserves_MW = 18
    ramp_rate_reactive_power_MVAr_per_min = 19
    APF = 20
