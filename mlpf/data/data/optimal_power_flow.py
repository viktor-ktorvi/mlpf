import torch

import numpy as np

from dataclasses import dataclass
from numpy import ndarray
from pypower.ppoption import ppoption
from pypower.runopf import runopf
from torch_geometric.data import Data

from mlpf.data.conversion.numpy.optimal_power_flow import ppc2optimal_power_flow_arrays
from mlpf.data.conversion.numpy.power_flow import ppc2power_flow_arrays
from mlpf.data.data.power_flow import PowerFlowData
from mlpf.data.masks.optimal_power_flow import create_optimal_power_flow_feature_mask
from mlpf.enumerations.bus_table import BusTableIds


@dataclass
class OptimalPowerFlowData(PowerFlowData):
    """
    A class that holds all the data needed for a machine learning optimal power flow.

    TODO individual field descriptions
    """
    baseMVA: float
    opf_features_matrix: ndarray
    target_cost: float
    total_feature_mask: ndarray

    def to_pyg_data(self, dtype: torch.dtype = torch.float32) -> Data:
        return Data(
            baseMVA=self.baseMVA,
            opf_features_matrix=torch.tensor(self.opf_features_matrix, dtype=dtype),
            target_cost=self.target_cost,
            total_feature_mask=torch.BoolTensor(self.total_feature_mask),
            PQVA_matrix=torch.tensor(self.PQVA_matrix, dtype=dtype),
            conductances_pu=torch.tensor(self.conductances_pu, dtype=dtype),
            edge_attr=torch.tensor(self.edge_attr, dtype=dtype),
            edge_index=torch.LongTensor(self.edge_index),
            feature_mask=torch.BoolTensor(self.feature_mask),
            feature_vector=torch.tensor(self.feature_vector, dtype=dtype).unsqueeze(0),
            susceptances_pu=torch.tensor(self.susceptances_pu, dtype=dtype),
            target_vector=torch.tensor(self.target_vector, dtype=dtype).unsqueeze(0),
            x=torch.tensor(self.x, dtype=dtype)
        )


def optimal_power_flow_data(ppc: dict, solve: bool = False, dtype: np.dtype = np.float64) -> OptimalPowerFlowData:
    """
    Extract all the data of the optimal power flow problem into a namespace object. The ppc should have a solved optimal power flow;
    otherwise if solve is set to True, and optimal power flow will be run.

    :param ppc: PYPOWER case object.
    :param solve: To run an OPF solver or not.
    :param dtype: Data type into which to cast the data.
    :return: OptimalPowerFlowData object.
    """
    if solve:
        ppc = runopf(ppc, ppopt=ppoption(OUT_ALL=0, VERBOSE=0))

    # extract data
    edge_index, active_powers_pu, reactive_powers_pu, voltages_pu, angles_rad, conductances_pu, susceptances_pu = ppc2power_flow_arrays(ppc, dtype=dtype)
    voltages_min, voltages_max, active_powers_min, active_powers_max, reactive_powers_min, reactive_powers_max, active_power_demands, reactive_power_demands, cost_coefficients = ppc2optimal_power_flow_arrays(
        ppc, dtype=dtype)

    PQVA_matrix = np.vstack((active_powers_pu, reactive_powers_pu, voltages_pu, angles_rad)).T

    opf_features_matrix = np.hstack((
        voltages_min.reshape(-1, 1),
        voltages_max.reshape(-1, 1),
        active_powers_min.reshape(-1, 1),
        active_powers_max.reshape(-1, 1),
        reactive_powers_min.reshape(-1, 1),
        reactive_powers_max.reshape(-1, 1),
        active_power_demands.reshape(-1, 1),
        reactive_power_demands.reshape(-1, 1),
        cost_coefficients
    ))

    feature_mask = create_optimal_power_flow_feature_mask(ppc["bus"][:, BusTableIds.bus_type])  # PQVA mask

    x = np.hstack((PQVA_matrix, opf_features_matrix))
    total_feature_mask = np.hstack((feature_mask, np.ones(shape=opf_features_matrix.shape, dtype=bool)))
    feature_vector = x[total_feature_mask]

    edge_attributes = np.vstack((conductances_pu, susceptances_pu))

    return OptimalPowerFlowData(
        PQVA_matrix=PQVA_matrix,
        baseMVA=ppc["baseMVA"],
        conductances_pu=conductances_pu,
        edge_attr=edge_attributes,
        edge_index=edge_index,
        feature_mask=feature_mask,
        feature_vector=feature_vector,
        opf_features_matrix=opf_features_matrix,
        susceptances_pu=susceptances_pu,
        target_cost=ppc["f"],
        target_vector=PQVA_matrix[~feature_mask],
        total_feature_mask=total_feature_mask,
        x=x
    )
