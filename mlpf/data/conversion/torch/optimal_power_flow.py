import torch

from torch import Tensor
from typing import Dict, Tuple

from mlpf.data.conversion.numpy.optimal_power_flow import ppc2optimal_power_flow_arrays


def ppc2optimal_power_flow_tensors(ppc: Dict, dtype: torch.dtype = torch.float64) -> Tuple[Tensor, ...]:
    """
    Extract the voltage and power limits, cost coefficients and power demands as arrays from the PYPOWER case object.
    The power demands are used to calculate the power generation given tensor of total power.

    Note that multiple generators per bus are currently not supported.

    :param ppc: PYPOWER case object.
    :param dtype: Torch data type.
    :return: (voltages_min, voltages_max, active_powers_min, active_powers_max, reactive_powers_min, reactive_powers_max, active_power_demands, reactive_power_demands, cost_coefficients)
    """

    voltages_min, voltages_max, \
        active_powers_min, active_powers_max, \
        reactive_powers_min, reactive_powers_max, \
        active_power_demands, reactive_power_demands, \
        cost_coefficients = ppc2optimal_power_flow_arrays(ppc)

    voltages_min = torch.tensor(voltages_min, dtype=dtype)
    voltages_max = torch.tensor(voltages_max, dtype=dtype)
    active_powers_min = torch.tensor(active_powers_min, dtype=dtype)
    active_powers_max = torch.tensor(active_powers_max, dtype=dtype)
    reactive_powers_min = torch.tensor(reactive_powers_min, dtype=dtype)
    reactive_powers_max = torch.tensor(reactive_powers_max, dtype=dtype)
    active_power_demands = torch.tensor(active_power_demands, dtype=dtype)
    reactive_power_demands = torch.tensor(reactive_power_demands, dtype=dtype)
    cost_coefficients = torch.tensor(cost_coefficients, dtype=dtype)

    return voltages_min, voltages_max, active_powers_min, active_powers_max, reactive_powers_min, reactive_powers_max, active_power_demands, reactive_power_demands, cost_coefficients
