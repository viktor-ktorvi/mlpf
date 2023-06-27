import torch

from torch import Tensor


def upper_bound_errors(value: Tensor, value_max: Tensor) -> Tensor:
    """
    Return the difference between the value and its maximum value if the value is larger than the maximum value, otherwise return 0.

    :param value:  Value.
    :param value_max: Maximum allowed value.
    :return: Error.
    """
    return torch.maximum(torch.zeros_like(value), value - value_max)


def lower_bound_errors(value: Tensor, value_min: Tensor) -> Tensor:
    """
    Return the difference between the value and its minimum value if the value is smaller than the minimum value, otherwise return 0.

    :param value: Value.
    :param value_min: Minimum allowed value.
    :return: Error.
    """
    return torch.minimum(torch.zeros_like(value), value - value_min)


def main():
    x = torch.linspace(-5, 5, 1000)
    offset = 0
    values = (torch.exp(-x ** 2) - 0.5) * 5 + offset
    upper_bound = torch.cos(x) + offset
    lower_bound = torch.sin(x) - 2 + offset

    upper_error = upper_bound_errors(values, upper_bound)
    lower_error = lower_bound_errors(values, lower_bound)
    upper_bound_mask = upper_error > 0
    lower_bound_mask = lower_error < 0

    alpha = 0.2

    fig, ax = plt.subplots(2, 1, sharex=True)
    fig.suptitle("Torch - Upper and lower bound error demonstration")
    ax[0].grid()
    ax[0].set_ylabel("value")
    ax[0].plot(x, values, color="b", label="value")
    ax[0].plot(x, upper_bound, color="crimson", linestyle="--", label="upper bound")
    ax[0].plot(x, lower_bound, color="orange", linestyle="--", label="lower bound")

    ax[0].fill_between(x, upper_bound, lower_bound, where=~(upper_bound_mask | lower_bound_mask), color="g", alpha=alpha, linewidth=0.0, label="correct")
    ax[0].fill_between(x, upper_bound, lower_bound, where=upper_bound_mask | lower_bound_mask, color="r", alpha=alpha, linewidth=0.0, label="incorrect")
    ax[0].legend()

    ax[1].grid()
    ax[1].set_ylabel("error")
    ax[1].plot(x, upper_error, color="crimson", label="upper error")
    ax[1].plot(x, lower_error, color="orange", label="lower error")
    ax[1].legend()

    plt.show()


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    main()
