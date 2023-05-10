import torch

from torch import Tensor
from torch_scatter import scatter_sum
from typing import Callable, Tuple


# TODO make numpy version for sklearn
def power_flow_errors(edge_index: Tensor,
                      active_powers: Tensor,
                      reactive_powers: Tensor,
                      voltages: Tensor,
                      angles: Tensor,
                      conductances: Tensor,
                      susceptances: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Calculate power flow equations errors. All the input variables need to be in SI units.

    :param edge_index: edge list
    :param active_powers: P
    :param reactive_powers: Q
    :param voltages: V magnitude
    :param angles: V angle(rad)
    :param conductances: G
    :param susceptances: B
    :return: active and reactive power errors
    """
    source = edge_index[0]
    target = edge_index[1]

    voltage_product = voltages[source] * voltages[target]

    angle_differences = angles[source] - angles[target]

    cosine = torch.cos(angle_differences)
    sine = torch.sin(angle_differences)

    active_power_losses = scatter_sum(voltage_product * conductances * cosine, source) + scatter_sum(voltage_product * susceptances * sine, source) - active_powers
    reactive_power_losses = scatter_sum(voltage_product * conductances * sine, source) - scatter_sum(voltage_product * susceptances * cosine, source) - reactive_powers

    return active_power_losses, reactive_power_losses


def power_flow_errors_pu(edge_index: Tensor,
                         active_powers_pu: Tensor,
                         reactive_powers_pu: Tensor,
                         voltages_pu: Tensor,
                         angles_deg: Tensor,
                         conductances_pu: Tensor,
                         susceptances_pu: Tensor,
                         baseMVA: float,
                         basekV: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Calculate power flow errors per unit(pu). All values must be provided per unit, except for angles which must be provided in
    degrees as is the standard in pypower and pandapower. The baseMVA scalar and the basekV tensor(for multi-voltage support)
    must be provided.

    :param edge_index: edge list
    :param active_powers_pu: P
    :param reactive_powers_pu: Q
    :param voltages_pu: V magnitude
    :param angles_deg: V angle
    :param conductances_pu: G
    :param susceptances_pu: B
    :param baseMVA: base power scalar in MVA
    :param basekV: base voltage tensor in kV
    :return:
    """
    active_powers = active_powers_pu * baseMVA
    reactive_powers = reactive_powers_pu * baseMVA

    voltages = voltages_pu * basekV
    angles = torch.deg2rad(angles_deg)

    base_admittance = baseMVA / basekV[edge_index[0]] / basekV[edge_index[1]]

    conductances = conductances_pu * base_admittance
    susceptances = susceptances_pu * base_admittance

    active_power_losses, reactive_power_losses = power_flow_errors(edge_index,
                                                                   active_powers,
                                                                   reactive_powers,
                                                                   voltages,
                                                                   angles,
                                                                   conductances,
                                                                   susceptances)

    active_power_losses_pu = active_power_losses / baseMVA
    reactive_power_losses_pu = reactive_power_losses / baseMVA

    return active_power_losses_pu, reactive_power_losses_pu


def scalarize(active_power_losses: Tensor,
              reactive_power_losses: Tensor,
              element_wise_function: Callable = torch.abs,
              active_power_coeff: float = 1.0,
              reactive_power_coeff: float = 1.0) -> Tensor:
    """
    Scalarize the tensors of errors by doing a weighted sum. An element wise operation can be applied to the tensor before
    summing up e.g. absolute value or square.

    :param active_power_losses: delta P
    :param reactive_power_losses: delta Q
    :param element_wise_function: callable
    :param active_power_coeff: P weight
    :param reactive_power_coeff: Q weight
    :return: scalar loss value
    """
    return torch.sum(
        active_power_coeff * element_wise_function(active_power_losses) +
        reactive_power_coeff * element_wise_function(reactive_power_losses)
    )
