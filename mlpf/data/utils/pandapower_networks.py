import pandapower.networks as pn
from typing import Any, List


def get_pp_example_networks() -> List[Any]:
    return [
        # pn.example_simple(),  # TODO size mismatch for this grid
        # pn.example_multivoltage(), # TODO size mismatch for this grid

    ]


def get_pp_test_networks() -> List[Any]:
    return [
        pn.panda_four_load_branch(),
        pn.four_loads_with_branches_out(),
        pn.simple_four_bus_system(),
        pn.simple_mv_open_ring_net(),
    ]


def get_cigre_networks() -> List[Any]:
    return [
        pn.create_cigre_network_hv(length_km_6a_6b=0.1),
        pn.create_cigre_network_mv(with_der=False),
        pn.create_cigre_network_mv(with_der="pv_wind"),
        pn.create_cigre_network_mv(with_der="all"),
        # pn.create_cigre_network_lv(), # TODO size mismatch for this grid

    ]


def get_kerber_networks() -> List[Any]:
    return [
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


def get_mv_oberrhein_networks() -> List[Any]:
    return [
        pn.mv_oberrhein()
    ]


def get_power_system_test_cases_networks() -> List[Any]:
    return [
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


def get_synthetic_voltage_control_lv_networks() -> List[Any]:
    return [
        pn.create_synthetic_voltage_control_lv_network(network_class='rural_1'),
        pn.create_synthetic_voltage_control_lv_network(network_class='rural_2'),
        pn.create_synthetic_voltage_control_lv_network(network_class='village_1'),
        pn.create_synthetic_voltage_control_lv_network(network_class='village_2'),
        pn.create_synthetic_voltage_control_lv_network(network_class='suburb_1'),

    ]


def get_dickert_lv_networks() -> List[Any]:
    return [
        pn.create_dickert_lv_network()
    ]


def get_three_phase_networks() -> List[Any]:
    return [
        pn.ieee_european_lv_asymmetric()
    ]


def get_all_pandapower_networks() -> List[Any]:
    return get_pp_example_networks() + \
        get_pp_test_networks() + \
        get_cigre_networks() + \
        get_kerber_networks() + \
        get_mv_oberrhein_networks() + \
        get_power_system_test_cases_networks() + \
        get_synthetic_voltage_control_lv_networks() + \
        get_dickert_lv_networks() + \
        get_three_phase_networks()
