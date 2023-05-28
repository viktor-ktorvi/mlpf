import unittest
import warnings
from typing import Dict

import pandapower as pp
import torch

from mlpf.data.utils.conversion import ppc2power_flow_tensors
from mlpf.data.utils.pandapower_networks import get_all_pandapower_networks
from mlpf.loss.power_flow import power_flow_errors, scalarize


def get_power_flow_loss(ppc: Dict, method="scatter", dtype: torch.dtype = torch.float64) -> float:
    """
    Get power flow loss from a ppc.

    :param ppc: pypower case format
    :param method: To use scatter operations on arrays or to multiply with sparse matrices.
    :param dtype: torch data type
    :return: loss
    """
    edge_index, active_powers_pu, reactive_powers_pu, voltages_pu, angles_rad, conductances_pu, susceptances_pu = ppc2power_flow_tensors(ppc, dtype)

    active_power_losses_pu, reactive_power_losses_pu = power_flow_errors(
        edge_index,
        active_powers_pu,
        reactive_powers_pu,
        voltages_pu, angles_rad,
        conductances_pu,
        susceptances_pu,
        method=method
    )

    return float(scalarize(active_power_losses_pu, reactive_power_losses_pu))


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
            self.assertAlmostEqual(get_power_flow_loss(net._ppc, method="scatter"), 0.0, places=places)
            self.assertAlmostEqual(get_power_flow_loss(net._ppc, method="sparse"), 0.0, places=places)


if __name__ == '__main__':
    unittest.main()
