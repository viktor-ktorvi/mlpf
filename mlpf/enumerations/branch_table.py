from enum import IntEnum


class BranchTableIds(IntEnum):
    """
    Enumeration for the indices of the ppc branch table.
    """
    from_bus_number = 0
    to_bus_number = 1
    resistance_pu = 2
    reactance_pu = 3
    total_line_charging_susceptance = 4
    rating_A_long_term_MVA = 5
    rating_B_short_term_MVA = 6
    rating_C_emergency_MVA = 7
    ratio = 8
    angle_deg = 9
    init_status = 10
    min_angle_diff_deg = 11
    max_angle_diff_deg = 12
