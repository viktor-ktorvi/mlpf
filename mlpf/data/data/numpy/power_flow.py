import numpy as np

from dataclasses import dataclass
from numpy import ndarray

from mlpf.data.conversion.numpy.power_flow import ppc2power_flow_arrays
from mlpf.data.masks.power_flow import create_power_flow_feature_mask
from mlpf.enumerations.bus_table import BusTableIds
from mlpf.utils.ppc import ppc_runpf


@dataclass
class PowerFlowData:
    """
    A class that holds all the data needed for a machine learning power flow.

    TODO individual field descriptions
    """
    PQVA_matrix: ndarray
    conductances_pu: ndarray
    edge_attr: ndarray
    edge_index: ndarray
    feature_mask: ndarray
    feature_vector: ndarray
    susceptances_pu: ndarray
    target_vector: ndarray
    x: ndarray


def power_flow_data(ppc: dict, solve: bool = False, dtype: np.dtype = np.float64) -> PowerFlowData:
    """
    Extract all the relevant info from the ppc file as ndarrays and pack it into a PyG Data object.

    :param ppc: PyPower case format object
    :param solve: If True, a power flow calculation in PyPower will be called before extracting info.
    :param dtype: Torch data type to cast the real valued tensors into.
    :return: PowerFlowData object.
    """
    if solve:
        ppc = ppc_runpf(ppc)

    edge_index, active_powers_pu, reactive_powers_pu, voltages_pu, angles_rad, conductances_pu, susceptances_pu = ppc2power_flow_arrays(ppc, dtype=dtype)

    PQVA_matrix = np.vstack((active_powers_pu, reactive_powers_pu, voltages_pu, angles_rad)).T
    feature_mask = create_power_flow_feature_mask(ppc["bus"][:, BusTableIds.bus_type])

    edge_attributes = np.vstack((conductances_pu, susceptances_pu))

    return PowerFlowData(
        PQVA_matrix=PQVA_matrix,
        conductances_pu=conductances_pu,
        edge_attr=edge_attributes,
        edge_index=edge_index,
        feature_mask=feature_mask,
        feature_vector=PQVA_matrix[feature_mask],
        susceptances_pu=susceptances_pu,
        target_vector=PQVA_matrix[~feature_mask],
        x=PQVA_matrix
    )
