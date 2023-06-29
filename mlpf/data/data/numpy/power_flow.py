import numpy as np

from types import SimpleNamespace

from mlpf.data.conversion.numpy.power_flow import ppc2power_flow_arrays
from mlpf.data.masks.power_flow import create_power_flow_feature_mask
from mlpf.enumerations.bus_table import BusTableIds
from mlpf.utils.ppc import ppc_runpf


def power_flow_data(ppc: dict, solve: bool = False, dtype: np.dtype = np.float64):
    """
    Extract all the relevant info from the ppc file as ndarrays and pack it into a PyG Data object.

    :param ppc: PyPower case format object
    :param solve: If True, a power flow calculation in PyPower will be called before extracting info.
    :param dtype: Torch data type to cast the real valued tensors into.
    """
    if solve:
        ppc = ppc_runpf(ppc)

    edge_index, active_powers_pu, reactive_powers_pu, voltages_pu, angles_rad, conductances_pu, susceptances_pu = ppc2power_flow_arrays(ppc, dtype=dtype)

    PQVA_matrix = np.vstack((active_powers_pu, reactive_powers_pu, voltages_pu, angles_rad)).T
    feature_mask = create_power_flow_feature_mask(ppc["bus"][:, BusTableIds.bus_type])

    edge_attributes = np.vstack((conductances_pu, susceptances_pu))

    return SimpleNamespace(
        edge_index=edge_index,
        x=PQVA_matrix,
        edge_attr=edge_attributes,
        PQVA_matrix=PQVA_matrix,
        feature_mask=feature_mask,
        conductances_pu=conductances_pu,
        susceptances_pu=susceptances_pu,
        feature_vector=PQVA_matrix[feature_mask],
        target_vector=PQVA_matrix[~feature_mask]
    )
