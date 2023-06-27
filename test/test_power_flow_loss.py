import unittest
import warnings
from typing import Dict

import numpy as np
import pandapower as pp
import torch
from pypower.ppoption import ppoption
from pypower.runpf import runpf

from mlpf.data.conversion.numpy.power_flow import ppc2power_flow_arrays
from mlpf.data.conversion.torch.power_flow import ppc2power_flow_tensors
from mlpf.data.utils.pandapower_networks import get_all_pandapower_networks
from mlpf.loss.numpy import power_flow as pf_numpy
from mlpf.loss.torch import power_flow as pf_torch


def torch_get_power_flow_loss(ppc: Dict, dtype: torch.dtype = torch.float64) -> float:
    """
    Get power flow loss from a ppc.

    :param ppc: pypower case format
    :param dtype: torch data type
    :return: loss
    """
    edge_index, active_powers_pu, reactive_powers_pu, voltages_pu, angles_rad, conductances_pu, susceptances_pu = ppc2power_flow_tensors(ppc, dtype)

    active_errors = pf_torch.active_power_errors(edge_index, active_powers_pu, voltages_pu, angles_rad, conductances_pu, susceptances_pu)
    reactive_errors = pf_torch.reactive_power_errors(edge_index, reactive_powers_pu, voltages_pu, angles_rad, conductances_pu, susceptances_pu)

    return float(torch.sum(active_errors + reactive_errors))


def numpy_get_power_flow_loss(ppc: Dict) -> float:
    """
    Get power flow loss from a ppc.

    :param ppc: pypower case format
    :return: loss
    """
    edge_index, active_powers_pu, reactive_powers_pu, voltages_pu, angles_rad, conductances_pu, susceptances_pu = ppc2power_flow_arrays(ppc)

    active_errors = pf_numpy.active_power_errors(edge_index, active_powers_pu, voltages_pu, angles_rad, conductances_pu, susceptances_pu)
    reactive_errors = pf_numpy.reactive_power_errors(edge_index, reactive_powers_pu, voltages_pu, angles_rad, conductances_pu, susceptances_pu)

    return float(np.sum(active_errors + reactive_errors))


class TestPowerFlowLoss(unittest.TestCase):
    def test_many_topologies(self):
        """
        Test for various grids if the power flow loss function calculates near 0 loss for solved ppcs in float64 accuracy.
        :return:
        """
        warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")

        nets = get_all_pandapower_networks()

        tolerance_VA = 1  # P error + Q error on entire grid in Volt-Amps

        for net in nets:
            print(net)

            ppc = pp.converter.to_ppc(net, init="flat")
            ppc, converged = runpf(ppc, ppopt=ppoption(OUT_ALL=0, VERBOSE=0))

            tolerance = tolerance_VA / ppc["baseMVA"]

            self.assertLess(torch_get_power_flow_loss(ppc), tolerance)
            self.assertLess(numpy_get_power_flow_loss(ppc), tolerance)


if __name__ == '__main__':
    unittest.main()
