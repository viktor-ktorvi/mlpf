import torch
import unittest
import warnings

import numpy as np
import pandapower as pp
import pandapower.networks as pn

from pypower.ppoption import ppoption
from pypower.runpf import runpf
from typing import Dict

from mlpf.data.conversion.numpy.power_flow import ppc2power_flow_arrays
from mlpf.data.conversion.torch.power_flow import ppc2power_flow_tensors
from mlpf.data.generate.generate_uniform_data import generate_uniform_ppcs
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
        Test for various grids if the power flow loss function calculates the loss bellow a certain threshold for solved ppcs in float64 accuracy.
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

    def test_case118_uniform(self):
        """
        Test for single topology but with random grid parameter values if the power flow loss function calculates the loss bellow a certain threshold for solved ppcs in float64 accuracy.
        :return:
        """
        warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")
        net = pn.case118()
        base_ppc = pp.converter.to_ppc(net, init="flat")

        base_ppc, converged = runpf(base_ppc, ppopt=ppoption(OUT_ALL=0, VERBOSE=0))

        ppc_list = generate_uniform_ppcs(
            base_ppc,
            how_many=1000,
            low=0.9,
            high=1.1
        )

        tolerance_VA = 1  # P error + Q error on entire grid in Volt-Amps
        tolerance = tolerance_VA / base_ppc["baseMVA"]

        for ppc in ppc_list:
            self.assertLess(torch_get_power_flow_loss(ppc), tolerance)
            self.assertLess(numpy_get_power_flow_loss(ppc), tolerance)


if __name__ == '__main__':
    unittest.main()
