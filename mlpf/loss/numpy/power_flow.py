from typing import Tuple, Callable

import numpy as np
from numpy import ndarray

from mlpf.loss.numpy.utils import make_sparse_admittance_matrix


def power_flow_errors(edge_index: ndarray,
                      active_powers: ndarray,
                      reactive_powers: ndarray,
                      voltages: ndarray,
                      angles_rad: ndarray,
                      conductances: ndarray,
                      susceptances: ndarray,
                      method: str = "sparse") -> Tuple[ndarray, ndarray]:
    """
    Calculate power flow equations errors efficiently. The values can be in either SI or per unit
    as long as the conventions are not mixed. Angles need to be in radians.

    :param edge_index: edge list
    :param active_powers: P
    :param reactive_powers: Q
    :param voltages: V magnitude
    :param angles_rad: V angle(rad)
    :param conductances: G
    :param susceptances: B
    :param method: How to perform the calculation.  # TODO more about the differences between the methods
    :return: active and reactive power errors
    """
    if method == "sparse":
        return power_flow_errors_sparse(edge_index, active_powers, reactive_powers, voltages, angles_rad, conductances, susceptances)

    raise ValueError(f"Method '{method}' is not supported. Supported methods are 'scatter' and 'sparse'")


def power_flow_errors_sparse(edge_index: ndarray,
                             active_powers: ndarray,
                             reactive_powers: ndarray,
                             voltages: ndarray,
                             angles_rad: ndarray,
                             conductances: ndarray,
                             susceptances: ndarray) -> Tuple[ndarray, ndarray]:
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

    complex_voltages = (voltages * np.exp(1j * angles_rad)).reshape(-1, 1)

    currents = admittance_matrix @ complex_voltages

    complex_powers = complex_voltages * np.conj(currents)

    complex_errors = complex_powers.squeeze() - (active_powers + 1j * reactive_powers)

    return np.real(complex_errors), np.imag(complex_errors)


def scalarize(active_power_errors: ndarray,
              reactive_power_errors: ndarray,
              element_wise_function: Callable = np.abs,
              active_power_coeff: float = 1.0,
              reactive_power_coeff: float = 1.0) -> ndarray:
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
    return np.sum(
        active_power_coeff * element_wise_function(active_power_errors) +
        reactive_power_coeff * element_wise_function(reactive_power_errors)
    )
