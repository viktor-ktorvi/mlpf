import torch

import numpy as np

from pypower.makeSbus import makeSbus
from pypower.makeYbus import makeYbus
from torch import LongTensor, Tensor
from typing import Dict, Tuple

from mlpf.enumerations.bus_table import BusTableIds


def extract_values(ppc: Dict, dtype: torch.dtype = torch.float32) -> Tuple[LongTensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor]:
    """
    Extract the physical values(needed for power flow) from a network and return them as torch tensors.

    :param ppc: pypower case format dict
    :param dtype: torch data type
    :return: edge list plus the physical values of a network
    """
    # extract powers - convert to per unit and return generation minus demand
    complex_power = makeSbus(ppc['baseMVA'], ppc['bus'], ppc['gen'])

    active_powers_pu = np.real(complex_power)
    reactive_powers_pu = np.imag(complex_power)

    # extract voltages and angles
    voltages_pu = ppc['bus'][:, BusTableIds.voltage_magnitude_pu]
    angles_deg = ppc['bus'][:, BusTableIds.voltage_angle_deg]

    # extract edges
    Y_sparse_matrix, _, _ = makeYbus(ppc['baseMVA'], ppc['bus'], ppc['branch'])

    source, target = Y_sparse_matrix.nonzero()
    line_admittances = np.array(Y_sparse_matrix[source, target]).squeeze()

    conductances_pu = np.real(line_admittances)
    susceptances_pu = np.imag(line_admittances)

    baseMVA = ppc['baseMVA']
    basekV = ppc['bus'][:, BusTableIds.base_kV]

    edge_index = np.vstack((source, target))

    # to tensor
    active_powers_pu = torch.tensor(active_powers_pu, dtype=dtype)
    reactive_powers_pu = torch.tensor(reactive_powers_pu, dtype=dtype)

    voltages_pu = torch.tensor(voltages_pu, dtype=dtype)
    angles_deg = torch.tensor(angles_deg, dtype=dtype)

    conductances_pu = torch.tensor(conductances_pu, dtype=dtype)
    susceptances_pu = torch.tensor(susceptances_pu, dtype=dtype)

    basekV = torch.tensor(basekV, dtype=dtype)

    edge_index = torch.LongTensor(edge_index)

    return edge_index, active_powers_pu, reactive_powers_pu, voltages_pu, angles_deg, conductances_pu, susceptances_pu, baseMVA, basekV
