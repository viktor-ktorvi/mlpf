import numpy as np

from numpy import ndarray


def polynomial_costs(power_generation: ndarray, cost_coefficients: ndarray) -> ndarray:
    """
    Calculate the polynomial costs of the power generation.

    TODO Latex
    :param power_generation: Array of powers.
    :param cost_coefficients: Matrix of polynomial coefficients.
    :return:
    """
    if len(power_generation.shape) == 1:
        power_generation = power_generation.reshape(-1, 1)

    num_coefficients = cost_coefficients.shape[1]  # n + 1
    polynomial_values = power_generation ** np.arange(num_coefficients)[::-1]  # p^n, p^(n-1),...,p^1, p^0

    # for 0 values in generator_powers, numpy says 0**0 = 1; so we need to set that to 0 explicitly
    nonzero_mask = power_generation != 0.0
    polynomial_values *= nonzero_mask

    return np.sum(polynomial_values * cost_coefficients, axis=1)
