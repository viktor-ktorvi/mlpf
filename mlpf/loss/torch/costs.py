import torch

from torch import Tensor


def polynomial_costs(power_generation: Tensor, cost_coefficients: Tensor) -> Tensor:
    """
    Calculate the polynomial costs of the power generation.

    TODO Latex
    :param power_generation: Tensor of powers.
    :param cost_coefficients: Matrix of polynomial coefficients.
    :return:
    """
    if len(power_generation.shape) == 1:
        power_generation = power_generation.reshape(-1, 1)

    num_coefficients = cost_coefficients.shape[1]  # n + 1
    polynomial_values = power_generation ** torch.arange(start=num_coefficients - 1, end=-1, step=-1, device=power_generation.device)  # p^n, p^(n-1),...,p^1, p^0

    # for 0 values in generator_powers, numpy says 0**0 = 1; so we need to set that to 0 explicitly
    nonzero_mask = power_generation != 0.0
    polynomial_values *= nonzero_mask

    return torch.sum(polynomial_values * cost_coefficients, dim=1)
