import copy
import warnings
from typing import Dict, List

import numpy as np
from numpy import ndarray
from pypower.ppoption import ppoption
from pypower.runpf import runpf
from tqdm import tqdm

from mlpf.enumerations.bus_table import BusTableIds
from mlpf.enumerations.generator_table import GeneratorTableIds


def generate_uniform_ppcs(base_ppc: Dict, how_many: int, low: float = 0.9, high: float = 1.1) -> List[Dict]:
    """
    Generate a list of ppcs for which the values of bus[active power, reactive power, voltage magnitude, voltage angle]
    and gen[active power, reactive power] is uniformly distributed around those same values in base_ppc. The ppcs come with a solved power flow.

    TODO add latex value~U(low*base_value, high*base_value)


    :param base_ppc: Base ppc around which to generate.
    :param how_many: How many ppcs to generate.
    :param low: Low value multiplier in the uniform distribution.
    :param high: High value multiplier in the uniform distribution.
    :return: List of ppcs
    """
    warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")

    bus_variables = [BusTableIds.active_power_MW, BusTableIds.reactive_power_MVAr, BusTableIds.voltage_magnitude_pu, BusTableIds.voltage_angle_deg]
    gen_variables = [GeneratorTableIds.active_power_MW, GeneratorTableIds.reactive_power_MVAr]

    class UniformDistribution:
        def __init__(self, low: ndarray, high: ndarray):
            self.low = low
            self.high = high

        def sample(self, size: int) -> ndarray:
            return np.random.rand(size) * (self.high - self.low) + self.low

    bus_distribution_parameters = {}  # set bus vars distributions
    for bus_variable in bus_variables:
        bus_distribution_parameters[bus_variable] = UniformDistribution(
            low=base_ppc["bus"][:, bus_variable] * low,
            high=base_ppc["bus"][:, bus_variable] * high
        )

    gen_distribution_parameters = {}  # set gen vars distributions
    for gen_variable in gen_variables:
        gen_distribution_parameters[gen_variable] = UniformDistribution(
            low=base_ppc["gen"][:, gen_variable] * low,
            high=base_ppc["gen"][:, gen_variable] * high
        )

    # sample from the distributions
    random_ppc_list = []
    for _ in tqdm(range(how_many), ascii=True, desc="Generating uniformly random ppc data"):

        converged = False
        solved_random_ppc = None

        # TODO this resampling might be skewing the distribution. This is telling me that I really need the test for this
        # repeat until convergence
        while not converged:
            # create a random ppc and solve the power flow of that ppcs
            random_ppc = copy.deepcopy(base_ppc)

            for bus_variable in bus_variables:
                random_ppc["bus"][:, bus_variable] = bus_distribution_parameters[bus_variable].sample(random_ppc["bus"].shape[0])

            for gen_variable in gen_variables:
                random_ppc["gen"][:, gen_variable] = gen_distribution_parameters[gen_variable].sample(random_ppc["gen"].shape[0])

            ppopt = ppoption(OUT_ALL=0, VERBOSE=0)
            solved_random_ppc, converged = runpf(random_ppc, ppopt=ppopt)

        random_ppc_list.append(solved_random_ppc)

    return random_ppc_list

# TODO test to see if the distribution really is uniform with the mean being the original value
