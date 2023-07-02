import torch
import unittest
import warnings

import numpy as np
import pandapower as pp
import pandapower.networks as pn

from pypower.ppoption import ppoption
from pypower.runopf import runopf
from pypower.runpf import runpf
from tqdm import tqdm

from mlpf.data.conversion.numpy.optimal_power_flow import ppc2optimal_power_flow_arrays
from mlpf.data.conversion.numpy.power_flow import ppc2power_flow_arrays
from mlpf.data.conversion.torch.optimal_power_flow import ppc2optimal_power_flow_tensors
from mlpf.data.conversion.torch.power_flow import ppc2power_flow_tensors
from mlpf.data.generate.generate_uniform_data import generate_uniform_ppcs
from mlpf.data.utils.pandapower_networks import get_all_pandapower_networks
from mlpf.enumerations.gencost_table import GeneratorCostTableIds
from mlpf.loss.numpy import bound_errors as bounds_numpy
from mlpf.loss.numpy.costs import polynomial_costs as polynomial_costs_numpy
from mlpf.loss.torch import bound_errors as bounds_torch
from mlpf.loss.torch.costs import polynomial_costs as polynomial_costs_torch


def get_bound_errors_numpy(ppc):
    edge_index, active_powers, reactive_powers, voltages, angles_rad, conductances, susceptances = ppc2power_flow_arrays(ppc)
    voltages_min, voltages_max, active_powers_min, active_powers_max, reactive_powers_min, reactive_powers_max, active_power_demands, reactive_power_demands, cost_coefficients = ppc2optimal_power_flow_arrays(
        ppc)

    voltage_upper_errors = bounds_numpy.upper_bound_errors(voltages, voltages_max)
    voltage_lower_errors = bounds_numpy.lower_bound_errors(voltages, voltages_min)

    active_upper_errors = bounds_numpy.upper_bound_errors(active_powers, active_powers_max)
    active_lower_errors = bounds_numpy.lower_bound_errors(active_powers, active_powers_min)

    reactive_upper_errors = bounds_numpy.upper_bound_errors(reactive_powers, reactive_powers_max)
    reactive_lower_errors = bounds_numpy.lower_bound_errors(reactive_powers, reactive_powers_min)

    return np.sum(voltage_upper_errors +
                  voltage_lower_errors +
                  active_upper_errors +
                  active_lower_errors +
                  reactive_upper_errors +
                  reactive_lower_errors)


def get_cost_difference_numpy(ppc):
    edge_index, active_powers, reactive_powers, voltages, angles_rad, conductances, susceptances = ppc2power_flow_arrays(ppc)
    voltages_min, voltages_max, active_powers_min, active_powers_max, reactive_powers_min, reactive_powers_max, active_power_demands, reactive_power_demands, cost_coefficients = ppc2optimal_power_flow_arrays(
        ppc)

    baseMVA = ppc["baseMVA"]

    active_powers_generation = (active_powers + active_power_demands) * baseMVA

    active_power_costs = polynomial_costs_numpy(active_powers_generation, cost_coefficients)

    total_costs = np.sum(active_power_costs)

    return np.abs(total_costs - ppc['f'])


def get_bound_errors_torch(ppc):
    edge_index, active_powers, reactive_powers, voltages, angles_rad, conductances, susceptances = ppc2power_flow_tensors(ppc, dtype=torch.float64)
    voltages_min, voltages_max, active_powers_min, active_powers_max, reactive_powers_min, reactive_powers_max, active_power_demands, reactive_power_demands, cost_coefficients = ppc2optimal_power_flow_tensors(
        ppc, dtype=torch.float64)

    voltage_upper_errors = bounds_torch.upper_bound_errors(voltages, voltages_max)
    voltage_lower_errors = bounds_torch.lower_bound_errors(voltages, voltages_min)

    active_upper_errors = bounds_torch.upper_bound_errors(active_powers, active_powers_max)
    active_lower_errors = bounds_torch.lower_bound_errors(active_powers, active_powers_min)

    reactive_upper_errors = bounds_torch.upper_bound_errors(reactive_powers, reactive_powers_max)
    reactive_lower_errors = bounds_torch.lower_bound_errors(reactive_powers, reactive_powers_min)

    return torch.sum(voltage_upper_errors +
                     voltage_lower_errors +
                     active_upper_errors +
                     active_lower_errors +
                     reactive_upper_errors +
                     reactive_lower_errors)


def get_cost_difference_torch(ppc):
    edge_index, active_powers, reactive_powers, voltages, angles_rad, conductances, susceptances = ppc2power_flow_tensors(ppc, dtype=torch.float64)
    voltages_min, voltages_max, active_powers_min, active_powers_max, reactive_powers_min, reactive_powers_max, active_power_demands, reactive_power_demands, cost_coefficients = ppc2optimal_power_flow_tensors(
        ppc, dtype=torch.float64)

    baseMVA = ppc["baseMVA"]

    active_powers_generation = (active_powers + active_power_demands) * baseMVA

    active_power_costs = polynomial_costs_torch(active_powers_generation, cost_coefficients)

    total_costs = torch.sum(active_power_costs)

    return torch.abs(total_costs - ppc['f'])


class TestOptimalPowerFlowLoss(unittest.TestCase):

    def test_case118_uniform(self):
        """
        Test the polynomial cost function and power and voltage bounds on a single topology(case118) but uniformly random parameters, for equality with the PYPOWER solution.
        :return:
        """
        warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")
        net = pn.case118()
        base_ppc = pp.converter.to_ppc(net, init="flat")

        base_ppc, converged = runpf(base_ppc, ppopt=ppoption(OUT_ALL=0, VERBOSE=0))

        assert (base_ppc["gencost"][:, GeneratorCostTableIds.model] == 2).all()

        ppc_list = generate_uniform_ppcs(
            base_ppc,
            how_many=10,
            low=0.9,
            high=1.1
        )

        tolerance = 1e-8
        for ppc in tqdm(ppc_list, ascii=True, desc="Checking costs and bounds"):
            opf_ppc = runopf(ppc, ppopt=ppoption(OUT_ALL=0, VERBOSE=0))
            self.assertLess(get_cost_difference_numpy(opf_ppc), tolerance)
            self.assertLess(get_bound_errors_numpy(opf_ppc), tolerance)

            self.assertLess(get_cost_difference_torch(opf_ppc), tolerance)
            self.assertLess(get_bound_errors_torch(opf_ppc), tolerance)

    # TODO the PYPOWER opf fails for some reason
    # def test_polynomial_cost_multiple_topologies(self):
    #     """
    #     Test the polynomial cost function on multiple topologies for equality with the PYPOWER solution.
    #     :return:
    #     """
    #     warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")
    #
    #     nets = get_all_pandapower_networks()
    #
    #     tolerance = 1e-8
    #     for net in tqdm(nets, ascii=True, desc="Multiple topologies"):
    #         base_ppc = pp.converter.to_ppc(net, init="flat")
    #         opf_ppc = runopf(base_ppc, ppopt=ppoption(OUT_ALL=0, VERBOSE=0))
    #         self.assertLess(get_cost_difference(opf_ppc), tolerance)


if __name__ == '__main__':
    unittest.main()
