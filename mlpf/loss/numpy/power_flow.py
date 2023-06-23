import numpy as np

from numpy import ndarray


def _scatter_sum(values: ndarray,
                 indices: ndarray,
                 out_size: int):
    """
    Same functionality as _torch_scatter.scatter_sum_.

    :param values: Array to be summed up.
    :param indices: Indices which define which elements will be summed together.
    :param out_size: Size(len) of the output array.
    :return: Resulting array.
    """
    result = np.zeros((out_size,))
    np.add.at(result, indices, values)

    return result


def _equation_components(edge_index: ndarray,
                         voltages: ndarray,
                         angles_rad: ndarray):
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

    cosine = np.cos(angle_differences)
    sine = np.sin(angle_differences)

    return source, target, voltage_product, sine, cosine


def active_power_errors(edge_index: ndarray,
                        active_powers: ndarray,
                        voltages: ndarray,
                        angles_rad: ndarray,
                        conductances: ndarray,
                        susceptances: ndarray):
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

    out_size = active_powers.shape[0]
    active_errors = _scatter_sum(voltage_product * conductances * cosine, source, out_size=out_size) + \
                    _scatter_sum(voltage_product * susceptances * sine, source, out_size=out_size) - active_powers

    return active_errors


def reactive_power_errors(edge_index: ndarray,
                          reactive_powers: ndarray,
                          voltages: ndarray,
                          angles_rad: ndarray,
                          conductances: ndarray,
                          susceptances: ndarray):
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

    out_size = reactive_powers.shape[0]
    reactive_errors = _scatter_sum(voltage_product * conductances * sine, source, out_size=out_size) - \
                      _scatter_sum(voltage_product * susceptances * cosine, source, out_size=out_size) - reactive_powers

    return reactive_errors
