import numpy as np

from numpy import ndarray
from typing import Dict, Tuple

from mlpf.enumerations.bus_table import BusTableIds
from mlpf.enumerations.gencost_table import GeneratorCostTableIds
from mlpf.enumerations.generator_table import GeneratorTableIds


def _zero_padding(indices: ndarray, values: ndarray, size: int):
    """
    Insert values at indices and zeros elsewhere.

    :param indices: Indices array, same size as values.
    :param values: Values array, same size as indices.
    :param size: How big is the final array.
    :return: Zero padded values.
    """
    shape = list(values.shape)
    shape[0] = size
    shape = tuple(shape)

    zero_padded_values = np.zeros(shape)
    zero_padded_values[indices] = values

    return zero_padded_values


def ppc2optimal_power_flow_arrays(ppc: Dict, dtype: np.dtype = np.float64) -> Tuple[ndarray, ...]:
    """
    Extract the voltage and power limits, cost coefficients and power demands as arrays from the PYPOWER case object.
    The power demands are used to calculate the power generation given an array of total power.

    Note that multiple generators per bus are currently not supported.

    :param ppc: PYPOWER case object.
    :param dtype: NumPy data type.
    :return: (voltages_min, voltages_max, active_powers_min, active_powers_max, reactive_powers_min, reactive_powers_max, active_power_demands, reactive_power_demands, cost_coefficients)
    """
    if ppc["gen"].shape[0] != len(np.unique(ppc["gen"][:, GeneratorTableIds.bus_number])):
        raise NotImplementedError("Multiple generators per bus are not yet supported.")

    voltages_min = ppc["bus"][:, BusTableIds.voltage_min_pu]
    voltages_max = ppc["bus"][:, BusTableIds.voltage_max_pu]

    active_power_demands = ppc["bus"][:, BusTableIds.active_power_MW] / ppc["baseMVA"]
    reactive_power_demands = ppc["bus"][:, BusTableIds.reactive_power_MVAr] / ppc["baseMVA"]

    size = ppc["bus"].shape[0]

    # generation - demand

    gen_bus_numbers = ppc["gen"][:, GeneratorTableIds.bus_number].astype(int)

    active_powers_min = _zero_padding(gen_bus_numbers, ppc["gen"][:, GeneratorTableIds.min_active_power_MW], size=size) / ppc["baseMVA"] - active_power_demands
    active_powers_max = _zero_padding(gen_bus_numbers, ppc["gen"][:, GeneratorTableIds.max_active_power_MW], size=size) / ppc["baseMVA"] - active_power_demands

    reactive_powers_min = _zero_padding(gen_bus_numbers, ppc["gen"][:, GeneratorTableIds.min_reactive_power_MVAr], size=size) / ppc["baseMVA"] - reactive_power_demands
    reactive_powers_max = _zero_padding(gen_bus_numbers, ppc["gen"][:, GeneratorTableIds.max_reactive_power_MVAr], size=size) / ppc["baseMVA"] - reactive_power_demands

    # TODO only supports costs for active power
    cost_coefficients = _zero_padding(gen_bus_numbers, ppc["gencost"][:, GeneratorCostTableIds.coefficients_start:], size=size)

    assert (voltages_min <= voltages_max).all()
    assert (active_powers_min <= active_powers_max).all()
    assert (reactive_powers_min <= reactive_powers_max).all()

    return voltages_min.astype(dtype), \
        voltages_max.astype(dtype), \
        active_powers_min.astype(dtype), \
        active_powers_max.astype(dtype), \
        reactive_powers_min.astype(dtype), \
        reactive_powers_max.astype(dtype), \
        active_power_demands.astype(dtype), \
        reactive_power_demands.astype(dtype), \
        cost_coefficients.astype(dtype)
