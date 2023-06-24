from typing import Any

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor


def relative_values(numerator: Any, denominator: Any, eps: float = 1e-9, fill: bool = True):
    # TODO test for arrays and tensors. What happens when fill=False and denominator is all zeros
    """
    Return the relative value of the numerator with respect to the denominator, all while protecting against
    divisions by zero or close to zero values.

    :param numerator: Array/Tensor to be divided.
    :param denominator: Array/Tensor to divide with.
    :param eps: Absolute value tolerance for being marked as zero value in the denominator.
    :param fill: Whether to fill in the values where there would have been a division by zero or not. In those cases
    returns a 0 relative value.
    :return: Relative values.
    """
    if type(denominator) == ndarray:
        zero_denominator_mask = np.abs(denominator) < eps
    elif type(denominator) == Tensor:
        zero_denominator_mask = torch.abs(denominator) < eps
    else:
        raise TypeError(f"Expected types 'ndarray' or 'Tensor' but got {type(denominator)} instead.")

    if fill:
        denominator[zero_denominator_mask] = 1.0

        rel_vals = numerator / denominator
        rel_vals[zero_denominator_mask] = 0.0

        return rel_vals
    else:
        return numerator[~zero_denominator_mask] / denominator[~zero_denominator_mask]
