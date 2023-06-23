import unittest
import warnings
from typing import Dict

import numpy as np
import pandapower as pp
import torch

from mlpf.data.conversion.numpy.power_flow import ppc2power_flow_arrays
from mlpf.data.conversion.torch.power_flow import ppc2power_flow_tensors
from mlpf.data.utils.pandapower_networks import get_all_pandapower_networks
from mlpf.loss.numpy import power_flow as pf_numpy
from mlpf.loss.torch import power_flow as pf_torch


def torch_get_power_flow_loss(ppc: Dict, method="scatter", dtype: torch.dtype = torch.float64) -> float:
    """
    Get power flow loss from a ppc.

    :param ppc: pypower case format
    :param method: To use scatter operations on arrays or to multiply with sparse matrices.
    :param dtype: torch data type
    :return: loss
    """
    edge_index, active_powers_pu, reactive_powers_pu, voltages_pu, angles_rad, conductances_pu, susceptances_pu = ppc2power_flow_tensors(ppc, dtype)

    active_power_losses_pu, reactive_power_losses_pu = pf_torch.power_flow_errors(
        edge_index,
        active_powers_pu,
        reactive_powers_pu,
        voltages_pu, angles_rad,
        conductances_pu,
        susceptances_pu,
        method=method
    )

    return float(pf_torch.scalarize(active_power_losses_pu, reactive_power_losses_pu))


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
        places = 5
        nets = get_all_pandapower_networks()
        for net in nets:
            print(net)
            pp.runpp(net, tolerance_mva=1e-10, numba=False)
            self.assertAlmostEqual(torch_get_power_flow_loss(net._ppc, method="scatter"), 0.0, places=places)
            self.assertAlmostEqual(torch_get_power_flow_loss(net._ppc, method="sparse"), 0.0, places=places)

            self.assertAlmostEqual(numpy_get_power_flow_loss(net._ppc), 0.0, places=places)


if __name__ == '__main__':
    unittest.main()
