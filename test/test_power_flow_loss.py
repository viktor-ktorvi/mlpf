import torch
import unittest

import pandapower as pp

from typing import Dict

from mlpf.data.utils.conversion import ppc2power_flow_values
from mlpf.data.utils.pandapower_networks import get_all_pandapower_networks
from mlpf.loss.power_flow import power_flow_errors_pu, scalarize


def get_power_flow_loss(ppc: Dict, dtype: torch.dtype = torch.float64) -> float:
    """
    Get power flow loss from a ppc.

    :param ppc: pypower case format
    :param dtype: torch data type
    :return: loss
    """
    edge_index, active_powers_pu, reactive_powers_pu, voltages_pu, angles_deg, conductances_pu, susceptances_pu, baseMVA, basekV = ppc2power_flow_values(ppc, dtype)

    active_power_losses_pu, reactive_power_losses_pu = power_flow_errors_pu(edge_index,
                                                                            active_powers_pu,
                                                                            reactive_powers_pu,
                                                                            voltages_pu,
                                                                            angles_deg,
                                                                            conductances_pu,
                                                                            susceptances_pu,
                                                                            baseMVA,
                                                                            basekV)

    return float(scalarize(active_power_losses_pu, reactive_power_losses_pu))


class TestPowerFlowLoss(unittest.TestCase):
    def test_many_topologies(self):
        """
        Test for various grids if the power flow loss function calculates near 0 loss for solved ppcs in float64 accuracy.
        :return:
        """

        nets = get_all_pandapower_networks()
        for net in nets:
            print(net)
            pp.runpp(net, tolerance_mva=1e-10, numba=False)
            self.assertAlmostEqual(get_power_flow_loss(net._ppc), 0.0, places=5)


if __name__ == '__main__':
    unittest.main()
