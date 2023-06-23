import torch
from torch import Tensor

from torch_scatter import scatter_sum


def _equation_components(edge_index: Tensor,
                         voltages: Tensor,
                         angles_rad: Tensor):
    """
    Extract the values used in the power flow equations.

    :param edge_index:
    :param voltages:
    :param angles_rad:
    :return:
    """
    source = edge_index[0]
    target = edge_index[1]

    voltage_product = voltages[source] * voltages[target]

    angle_differences = angles_rad[source] - angles_rad[target]

    cosine = torch.cos(angle_differences)
    sine = torch.sin(angle_differences)

    return source, target, voltage_product, sine, cosine


def active_power_errors(edge_index: Tensor,
                        active_powers: Tensor,
                        voltages: Tensor,
                        angles_rad: Tensor,
                        conductances: Tensor,
                        susceptances: Tensor):
    """
    Active power error as defined by the power flow equations.

    TODO Latex

    :param edge_index: Edge list.
    :param active_powers: Active powers.
    :param voltages: Voltage magnitudes.
    :param angles_rad: Voltage angles.
    :param conductances: Conductances(real part of the admittance matrix elements).
    :param susceptances: Susceptances(imaginary part of the admittance matrix elements).
    :return: Array of active power errors.
    """

    source, target, voltage_product, sine, cosine = _equation_components(edge_index,
                                                                         voltages,
                                                                         angles_rad)

    active_errors = scatter_sum(voltage_product * conductances * cosine, source) + \
                    scatter_sum(voltage_product * susceptances * sine, source) - active_powers

    return active_errors


def reactive_power_errors(edge_index: Tensor,
                          reactive_powers: Tensor,
                          voltages: Tensor,
                          angles_rad: Tensor,
                          conductances: Tensor,
                          susceptances: Tensor):
    """
    Reactive power error as defined by the power flow equations.

    TODO Latex

    :param edge_index: Edge list.
    :param reactive_powers: Active powers.
    :param voltages: Voltage magnitudes.
    :param angles_rad: Voltage angles.
    :param conductances: Conductances(real part of the admittance matrix elements).
    :param susceptances: Susceptances(imaginary part of the admittance matrix elements).
    :return: Array of active power errors.
    """

    source, target, voltage_product, sine, cosine = _equation_components(edge_index,
                                                                         voltages,
                                                                         angles_rad)

    reactive_errors = scatter_sum(voltage_product * conductances * sine, source) - \
                      scatter_sum(voltage_product * susceptances * cosine, source) - reactive_powers

    return reactive_errors
