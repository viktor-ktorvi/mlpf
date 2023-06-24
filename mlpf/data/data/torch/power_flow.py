import torch

from torch_geometric.data import Data

from mlpf.data.conversion.torch.power_flow import ppc2power_flow_tensors
from mlpf.data.utils.masks import create_feature_mask
from mlpf.enumerations.bus_table import BusTableIds
from mlpf.utils.ppc import ppc_runpf


def power_flow_data(ppc: dict, solve: bool = False, dtype: torch.dtype = torch.float32):
    """
    Extract all the relevant info from the ppc file as tensors and pack it into a PyG Data object.

    :param ppc: PyPower case format object
    :param solve: If True, a power flow calculation in PyPower will be called before extracting info.
    :param dtype: Torch data type to cast the real valued tensors into.
    """
    if solve:
        ppc = ppc_runpf(ppc)

    edge_index, active_powers_pu, reactive_powers_pu, voltages_pu, angles_rad, conductances_pu, susceptances_pu = ppc2power_flow_tensors(ppc, dtype=dtype)

    PQVA_matrix = torch.vstack((active_powers_pu, reactive_powers_pu, voltages_pu, angles_rad)).T
    feature_mask = torch.BoolTensor(create_feature_mask(ppc["bus"][:, BusTableIds.bus_type]))

    edge_attributes = torch.vstack((conductances_pu, susceptances_pu))

    return Data(
        edge_index=edge_index,
        x=PQVA_matrix,
        edge_attr=edge_attributes,
        PQVA_matrix=PQVA_matrix,
        feature_mask=feature_mask,
        conductances_pu=conductances_pu,
        susceptances_pu=susceptances_pu,
        feature_vector=PQVA_matrix[feature_mask].unsqueeze(0),
        target_vector=PQVA_matrix[~feature_mask].unsqueeze(0)
    )
