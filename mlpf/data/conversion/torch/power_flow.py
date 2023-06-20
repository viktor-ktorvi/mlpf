from typing import Dict, Tuple

import torch
from torch import LongTensor, Tensor

from mlpf.data.conversion.numpy.power_flow import ppc2power_flow_arrays


def ppc2power_flow_tensors(ppc: Dict, dtype: torch.dtype = torch.float32) -> Tuple[LongTensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Extract the physical values(needed for power flow) from a network and return them as torch tensors.

    :param ppc: pypower case format dict
    :param dtype: torch data type
    :return: (edge_index, active_powers_pu, reactive_powers_pu, voltages_pu, angles_rad, conductances_pu, susceptances_pu)
    """

    edge_index, active_powers_pu, reactive_powers_pu, voltages_pu, angles_rad, conductances_pu, susceptances_pu = ppc2power_flow_arrays(ppc)

    # to tensor
    active_powers_pu = torch.tensor(active_powers_pu, dtype=dtype)
    reactive_powers_pu = torch.tensor(reactive_powers_pu, dtype=dtype)

    voltages_pu = torch.tensor(voltages_pu, dtype=dtype)
    angles_rad = torch.tensor(angles_rad, dtype=dtype)

    conductances_pu = torch.tensor(conductances_pu, dtype=dtype)
    susceptances_pu = torch.tensor(susceptances_pu, dtype=dtype)

    edge_index = torch.LongTensor(edge_index)

    return edge_index, active_powers_pu, reactive_powers_pu, voltages_pu, angles_rad, conductances_pu, susceptances_pu
