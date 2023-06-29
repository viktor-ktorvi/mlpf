from enum import IntEnum


class OptimalPowerFlowFeatureIds(IntEnum):
    """
    Enumeration for auxiliary OPF features.
    """
    voltages_min = 0,
    voltages_max = 1,
    active_powers_min = 2,
    active_powers_max = 3,
    reactive_powers_min = 4,
    reactive_powers_max = 5,
    active_power_demands = 6,
    reactive_power_demands = 7,
    cost_coefficients_start = 8
