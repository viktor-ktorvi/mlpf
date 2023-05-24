from enum import IntEnum


class PowerFlowFeatureIds(IntEnum):
    """
    Enumeration for data arranged as PQVA i.e. (active power, reactive_power, voltage magnitude, voltage angle).
    """
    active_power = 0
    reactive_power = 1
    voltage_magnitude = 2
    voltage_angle = 3
