import torch

from torch import Tensor
from torch_scatter import scatter_sum
from typing import Callable, Tuple

from mlpf.loss.utils import make_sparse_admittance_matrix


# TODO make numpy version for sklearn
def power_flow_errors_scatter(edge_index: Tensor,
                              active_powers: Tensor,
                              reactive_powers: Tensor,
                              voltages: Tensor,
                              angles_rad: Tensor,
                              conductances: Tensor,
                              susceptances: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Calculate power flow equations errors efficiently using scatter sums. The values can be in either SI or per unit
    as long as the conventions are not mixed.Angles need to be in radians.

    :param edge_index: edge list
    :param active_powers: P
    :param reactive_powers: Q
    :param voltages: V magnitude
    :param angles_rad: V angle(rad)
    :param conductances: G
    :param susceptances: B
    :return: active and reactive power errors
    """
    source = edge_index[0]
    target = edge_index[1]

    voltage_product = voltages[source] * voltages[target]

    angle_differences = angles_rad[source] - angles_rad[target]

    cosine = torch.cos(angle_differences)
    sine = torch.sin(angle_differences)

    active_power_errors = scatter_sum(voltage_product * conductances * cosine, source) + scatter_sum(voltage_product * susceptances * sine, source) - active_powers
    reactive_power_errors = scatter_sum(voltage_product * conductances * sine, source) - scatter_sum(voltage_product * susceptances * cosine, source) - reactive_powers

    return active_power_errors, reactive_power_errors


def power_flow_errors_sparse(edge_index: Tensor,
                             active_powers: Tensor,
                             reactive_powers: Tensor,
                             voltages: Tensor,
                             angles_rad: Tensor,
                             conductances: Tensor,
                             susceptances: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Calculate power flow equations errors efficiently using sparse matrix multiplication. The values can be in either SI or per unit
    as long as the conventions are not mixed. Angles need to be in radians.

    :param edge_index: edge list
    :param active_powers: P
    :param reactive_powers: Q
    :param voltages: V magnitude
    :param angles_rad: V angle(rad)
    :param conductances: G
    :param susceptances: B
    :return: active and reactive power errors
    """
    admittance_matrix = make_sparse_admittance_matrix(edge_index, conductances, susceptances)

    complex_voltages = (voltages * torch.exp(1j * angles_rad)).reshape(-1, 1)

    currents = admittance_matrix @ complex_voltages

    complex_powers = complex_voltages * torch.conj(currents)

    complex_errors = complex_powers.squeeze() - (active_powers + 1j * reactive_powers)

    return torch.real(complex_errors), torch.imag(complex_errors)


def scalarize(active_power_errors: Tensor,
              reactive_power_errors: Tensor,
              element_wise_function: Callable = torch.abs,
              active_power_coeff: float = 1.0,
              reactive_power_coeff: float = 1.0) -> Tensor:
    """
    Scalarize the tensors of errors by doing a weighted sum. An element wise operation can be applied to the tensor before
    summing up e.g. absolute value or square.

    :param active_power_errors: delta P
    :param reactive_power_errors: delta Q
    :param element_wise_function: callable
    :param active_power_coeff: P weight
    :param reactive_power_coeff: Q weight
    :return: scalar loss value
    """
    return torch.sum(
        active_power_coeff * element_wise_function(active_power_errors) +
        reactive_power_coeff * element_wise_function(reactive_power_errors)
    )
