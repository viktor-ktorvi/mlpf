import copy

import hydra

import numpy as np
import pandapower as pp
import pandapower.networks as pn

from numpy import ndarray
from tqdm import tqdm
from typing import Dict, List

from mlpf.data.utils.saving import pickle_all
from mlpf.enumerations.bus_table import BusTableIds
from mlpf.enumerations.generator_table import GeneratorTableIds


def generate_uniform_ppcs(base_ppc: Dict, how_many: int, low: float = 0.9, high: float = 1.1) -> List[Dict]:
    """
    Generate a list of ppcs for which the values of bus[active power, reactive power, voltage magnitude, voltage angle]
    and gen[active power, reactive power] is uniformly distributed around those same values in base_ppc.

    TODO add latex value~U(low*base_value, high*base_value)


    :param base_ppc: Base ppc around which to generate.
    :param how_many: How many ppcs to generate.
    :param low: Low value multiplier in the uniform distribution.
    :param high: High value multiplier in the uniform distribution.
    :return: List of ppcs
    """

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
        random_ppc = copy.deepcopy(base_ppc)

        for bus_variable in bus_variables:
            random_ppc["bus"][:, bus_variable] = bus_distribution_parameters[bus_variable].sample(random_ppc["bus"].shape[0])

        for gen_variable in gen_variables:
            random_ppc["gen"][:, gen_variable] = gen_distribution_parameters[gen_variable].sample(random_ppc["gen"].shape[0])

        random_ppc_list.append(random_ppc)

    return random_ppc_list


# TODO test to see if the distribution really is uniform with the mean being the original value
@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg):
    """
    Generate a dataset and save it. Use the hydra config default or overwrite the command line args.

    :param cfg: hydra config
    :return:
    """
    # net = pn.create_kerber_dorfnetz()  # TODO think of an elegant way to choose grids from the command line
    net = pn.case118()
    pp.runpp(net)
    base_ppc = net._ppc

    uniform_ppc_list = generate_uniform_ppcs(
        base_ppc,
        how_many=cfg.how_many,
        low=cfg.low,
        high=cfg.high
    )

    pickle_all(uniform_ppc_list, save_path=cfg.save_path, extension=cfg.extension, delete_all_from_save_path=True)


if __name__ == "__main__":
    main()
