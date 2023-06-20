import warnings
from typing import Dict, Tuple

import numpy as np
from numpy import ndarray
from pypower.makeSbus import makeSbus
from pypower.makeYbus import makeYbus

from mlpf.enumerations.bus_table import BusTableIds


def ppc2power_flow_arrays(ppc: Dict, dtype: np.dtype = np.float64) -> Tuple[ndarray, ...]:
    """
    Extract the physical values(needed for power flow) from a network

    :param ppc: PyPower case format dict.
    :param dtype: NumPy data type.
    :return: (edge_index, active_powers_pu, reactive_powers_pu, voltages_pu, angles_rad, conductances_pu, susceptances_pu)
    """
    # extract powers - convert to per unit and return generation minus demand
    complex_power = makeSbus(ppc['baseMVA'], ppc['bus'], ppc['gen'])

    active_powers_pu = np.real(complex_power).astype(dtype)
    reactive_powers_pu = np.imag(complex_power).astype(dtype)

    # extract voltages and angles
    voltages_pu = ppc['bus'][:, BusTableIds.voltage_magnitude_pu].astype(dtype)
    angles_deg = ppc['bus'][:, BusTableIds.voltage_angle_deg]
    angles_rad = np.deg2rad(angles_deg).astype(dtype)

    warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part")

    # extract edges
    Y_sparse_matrix, _, _ = makeYbus(ppc['baseMVA'], ppc['bus'], ppc['branch'])

    source, target = Y_sparse_matrix.nonzero()
    line_admittances = np.array(Y_sparse_matrix[source, target]).squeeze()

    conductances_pu = np.real(line_admittances).astype(dtype)
    susceptances_pu = np.imag(line_admittances).astype(dtype)

    edge_index = np.vstack((source, target)).astype(np.int64)

    return edge_index, active_powers_pu, reactive_powers_pu, voltages_pu, angles_rad, conductances_pu, susceptances_pu
