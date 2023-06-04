import pickle
import warnings
from typing import Dict

import numpy as np
from matplotlib import pyplot as plt
from pypower.makeYbus import makeYbus
from torch import Tensor
from tqdm import tqdm

from mlpf.data.loading.load_data import autodetect_load_ppc
from mlpf.enumerations.branch_table import BranchTableIds
from mlpf.enumerations.bus_table import BusTableIds


def line_currents_ppc(ppc: Dict) -> Tensor:
    _, Yf, _ = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])

    voltage_magnitudes_pu = ppc["bus"][:, BusTableIds.voltage_magnitude_pu]
    angles_rad = np.deg2rad(ppc["bus"][:, BusTableIds.voltage_angle_deg])

    complex_voltages_pu = voltage_magnitudes_pu * np.exp(1j * angles_rad)

    complex_currents = Yf @ complex_voltages_pu

    return complex_currents


def load_unsolved_solved_ppc_tuple(filepath: str):
    with open(filepath, 'rb') as f:
        unsolved_ppc, solved_ppc = pickle.load(f)

    return unsolved_ppc, solved_ppc


def currents_main():
    ppc_list = autodetect_load_ppc("generated_ppcs")
    # ppc_tuples = load_data("../eon_rl_data/LV_EON/20230328_OPF_lv_grid_403059_st_svl_400V_t35040",
    #                        load_sample_function=load_unsolved_solved_ppc_tuple,
    #                        shuffle=True,
    #                        max_samples=5000)
    #
    # ppc_list = [ppc_tuple[0] for ppc_tuple in ppc_tuples if type(ppc_tuple[1]) == dict]

    warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")
    relative_current_magnitudes_list = []

    not_converged_counter = 0
    for ppc in tqdm(ppc_list):
        current_magnitudes = np.abs(line_currents_ppc(ppc))

        current_upper_limits = ppc["branch"][:, BranchTableIds.rating_A_long_term_MVA]

        relative_current_magnitudes_list.append(current_magnitudes / current_upper_limits)

    print(f"Not converged count = {not_converged_counter}")
    relative_current_magnitudes = np.array(relative_current_magnitudes_list).flatten()
    plt.figure()
    plt.yscale("log")
    plt.hist(relative_current_magnitudes, bins=100)
    plt.axvline(x=1.0, color='r', linestyle="--")

    plt.show()
    kjkszpj = None


def main():
    kjkszpj = None


if __name__ == "__main__":
    main()
