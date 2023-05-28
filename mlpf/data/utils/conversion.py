import warnings
from typing import Dict, Tuple, List

import numpy as np
import pandapower as pp
import torch
from numpy import ndarray
from pandapower import pandapowerNet
from pypower.makeSbus import makeSbus
from pypower.makeYbus import makeYbus
from torch import LongTensor, Tensor

from mlpf.enumerations.bus_table import BusTableIds


def ppc2power_flow_arrays(ppc: Dict) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
    """
    Extract the physical values(needed for power flow) from a network

    :param ppc: pypower case format dict
    :return: (edge_index, active_powers_pu, reactive_powers_pu, voltages_pu, angles_rad, conductances_pu, susceptances_pu)
    """
    # extract powers - convert to per unit and return generation minus demand
    complex_power = makeSbus(ppc['baseMVA'], ppc['bus'], ppc['gen'])

    active_powers_pu = np.real(complex_power)
    reactive_powers_pu = np.imag(complex_power)

    # extract voltages and angles
    voltages_pu = ppc['bus'][:, BusTableIds.voltage_magnitude_pu]
    angles_deg = ppc['bus'][:, BusTableIds.voltage_angle_deg]
    angles_rad = np.deg2rad(angles_deg)

    warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")

    # extract edges
    Y_sparse_matrix, _, _ = makeYbus(ppc['baseMVA'], ppc['bus'], ppc['branch'])

    source, target = Y_sparse_matrix.nonzero()
    line_admittances = np.array(Y_sparse_matrix[source, target]).squeeze()

    conductances_pu = np.real(line_admittances)
    susceptances_pu = np.imag(line_admittances)

    edge_index = np.vstack((source, target))

    return edge_index, active_powers_pu, reactive_powers_pu, voltages_pu, angles_rad, conductances_pu, susceptances_pu


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


def pandapower2ppc_list(pandapower_networks: List[pandapowerNet]) -> List[Dict]:
    """
    Convert a list of pandapower networks to a list of pypower case files.

    :param pandapower_networks:
    :return:
    """
    ppc_list = []
    for net in pandapower_networks:
        pp.runpp(net, numba=False)
        ppc_list.append(net._ppc)

    return ppc_list
