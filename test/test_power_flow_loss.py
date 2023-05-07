import torch
import unittest

import pandapower as pp
import pandapower.networks as pn

from typing import Dict

from mlpf.data.utils.values_from_ppc import extract_values
from mlpf.loss.power_flow import power_flow_errors_pu, scalarize


def get_power_flow_loss(ppc: Dict, dtype: torch.dtype = torch.float64) -> float:
    """
    Get power flow loss from a ppc.

    :param ppc: pypower case format
    :param dtype: torch data type
    :return: loss
    """
    edge_index, active_powers_pu, reactive_powers_pu, voltages_pu, angles_deg, conductances_pu, susceptances_pu, baseMVA, basekV = extract_values(ppc, dtype)

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
    def test_power_flow_loss(self):
        """
        Test for various grids if the power flow loss function calculates near 0 loss for solved ppcs in float64 accuracy.
        :return:
        """

        pp_example_networks = [
            # pn.example_simple(),  # TODO size mismatch for this grid
            # pn.example_multivoltage(), # TODO size mismatch for this grid

        ]

        pp_test_networks = [
            pn.panda_four_load_branch(),
            pn.four_loads_with_branches_out(),
            pn.simple_four_bus_system(),
            pn.simple_mv_open_ring_net(),
        ]

        cigre_networks = [
            pn.create_cigre_network_hv(length_km_6a_6b=0.1),
            pn.create_cigre_network_mv(with_der=False),
            pn.create_cigre_network_mv(with_der="pv_wind"),
            pn.create_cigre_network_mv(with_der="all"),
            # pn.create_cigre_network_lv(), # TODO size mismatch for this grid

        ]

        kerber_networks = [
            pn.create_kerber_landnetz_freileitung_1(),
            pn.create_kerber_landnetz_freileitung_2(),
            pn.create_kerber_landnetz_kabel_1(),
            pn.create_kerber_landnetz_kabel_2(),
            pn.create_kerber_dorfnetz(),
            pn.create_kerber_vorstadtnetz_kabel_1(),
            pn.create_kerber_vorstadtnetz_kabel_2(),
            pn.kb_extrem_landnetz_freileitung(),
            pn.kb_extrem_landnetz_kabel(),
            pn.kb_extrem_landnetz_freileitung_trafo(),
            pn.kb_extrem_landnetz_kabel_trafo(),
            pn.kb_extrem_dorfnetz(),
            pn.kb_extrem_dorfnetz_trafo(),
            pn.kb_extrem_vorstadtnetz_1(),
            pn.kb_extrem_vorstadtnetz_2(),
            pn.kb_extrem_vorstadtnetz_trafo_1(),
            pn.kb_extrem_vorstadtnetz_trafo_2()
        ]

        mv_oberrhein_networks = [
            pn.mv_oberrhein()
        ]

        power_system_test_cases_networks = [
            pn.case4gs(),
            pn.case5(),
            pn.case6ww(),
            pn.case9(),
            pn.case14(),
            pn.case24_ieee_rts(),
            pn.case30(),
            pn.case_ieee30(),
            pn.case33bw(),
            pn.case39(),
            pn.case57(),
            pn.case89pegase(),
            pn.case118(),
            pn.case145(),
            pn.case_illinois200(),
            pn.case300(),
            pn.case1354pegase(),
            pn.case1888rte(),
            pn.case2848rte(),
            pn.case2869pegase(),
            pn.case3120sp(),
            pn.case6470rte(),
            pn.case6495rte(),
            pn.case6515rte(),
            pn.case9241pegase(),
            pn.GBnetwork(),
            pn.GBreducednetwork(),
            pn.iceland()
        ]

        synthetic_voltage_control_lv_networks = [
            pn.create_synthetic_voltage_control_lv_network(network_class='rural_1'),
            pn.create_synthetic_voltage_control_lv_network(network_class='rural_2'),
            pn.create_synthetic_voltage_control_lv_network(network_class='village_1'),
            pn.create_synthetic_voltage_control_lv_network(network_class='village_2'),
            pn.create_synthetic_voltage_control_lv_network(network_class='suburb_1'),

        ]

        dickert_lv_networks = [
            pn.create_dickert_lv_network()
        ]

        three_phase_networks = [
            pn.ieee_european_lv_asymmetric()
        ]

        nets = pp_example_networks + pp_test_networks + cigre_networks + kerber_networks + mv_oberrhein_networks + \
               power_system_test_cases_networks + synthetic_voltage_control_lv_networks + dickert_lv_networks + three_phase_networks
        for net in nets:
            print(net)
            pp.runpp(net, tolerance_mva=1e-10)
            self.assertAlmostEqual(get_power_flow_loss(net._ppc), 0.0, places=5)


if __name__ == '__main__':
    unittest.main()
