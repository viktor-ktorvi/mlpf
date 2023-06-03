from typing import List, Tuple

import numpy as np
from numpy import ndarray
from pypower.ppoption import ppoption
from pypower.runpf import runpf

from mlpf.data.utils.conversion import ppc2power_flow_arrays
from mlpf.data.utils.masks import create_feature_mask
from mlpf.enumerations.bus_table import BusTableIds
from mlpf.enumerations.power_flow_ids import PowerFlowFeatureIds


class PowerFlowData:
    """
    A class that holds all the relevant info for supervised and unsupervised power flow calculations.
    """

    def __init__(self, ppc: dict, solved: bool = True):
        """
        Extract all the relevant info from the ppc file.
        :param ppc: PyPower case format object
        :param solved: If False, a power flow calculation in PyPower will be called before extracting info.
        """
        if not solved:
            ppopt = ppoption(OUT_ALL=0, VERBOSE=0)
            ppc, converged = runpf(ppc, ppopt=ppopt)

            assert converged, "Power flow hasn't converged for a sample."

        edge_index, active_powers_pu, reactive_powers_pu, voltages_pu, angles_rad, conductances_pu, susceptances_pu = ppc2power_flow_arrays(ppc)

        grid_feature_matrix = np.vstack((active_powers_pu, reactive_powers_pu, voltages_pu, angles_rad)).T
        feature_mask = create_feature_mask(ppc["bus"][:, BusTableIds.bus_type])

        self.feature_mask = feature_mask
        self.conductances_pu = conductances_pu
        self.susceptances_pu = susceptances_pu
        self.edge_index = edge_index

        self.feature_vector = grid_feature_matrix[self.feature_mask]
        self.target_vector = grid_feature_matrix[~self.feature_mask]

        self.grid_feature_matrix = grid_feature_matrix

    @property
    def active_power(self):
        return self.grid_feature_matrix[:, PowerFlowFeatureIds.active_power]

    @property
    def reactive_power(self):
        return self.grid_feature_matrix[:, PowerFlowFeatureIds.reactive_power]

    @property
    def voltage_magnitude(self):
        return self.grid_feature_matrix[:, PowerFlowFeatureIds.voltage_magnitude]

    @property
    def voltage_angle(self):
        return self.grid_feature_matrix[:, PowerFlowFeatureIds.voltage_angle]


def data2features_and_targets(data_list: List[PowerFlowData]) -> Tuple[ndarray, ndarray]:
    """
    Concatenate the feature/target vectors in a list of data objects to get a feature/target matrix.
    :param data_list: List of data objects
    :return:
    """
    feature_matrix = np.vstack([data.feature_vector for data in data_list])
    target_matrix = np.vstack([data.target_vector for data in data_list])

    return feature_matrix, target_matrix
