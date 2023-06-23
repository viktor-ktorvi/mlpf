from typing import Tuple

import torch
from torch import Tensor
from torch_geometric.data import Data

from mlpf.data.conversion.torch.power_flow import ppc2power_flow_tensors
from mlpf.data.utils.masks import create_feature_mask
from mlpf.enumerations.bus_table import BusTableIds
from mlpf.enumerations.power_flow_ids import PowerFlowFeatureIds
from mlpf.loss.relative_values import relative_values
from mlpf.loss.torch.power_flow import active_power_errors, reactive_power_errors
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


def get_relative_power_flow_errors(predictions: Tensor, batch: Data) -> Tuple[Tensor, ...]:
    """
    Take model predictions and merge them with the corresponding data batch.
    Calculate the relative power flow errors for the given predictions.

    :param predictions: Power flow unknown variable predictions.
    :param batch: Data batch from which the predictions came from.
    :return: (relative active power errors, relative reactive power errors)
    """

    PQVA_matrix_prediction = batch.PQVA_matrix.detach().clone()  # deep copy
    PQVA_matrix_prediction[~batch.feature_mask] = predictions.flatten()

    active_powers = PQVA_matrix_prediction[:, PowerFlowFeatureIds.active_power]
    reactive_powers = PQVA_matrix_prediction[:, PowerFlowFeatureIds.reactive_power]

    voltage_magnitudes = PQVA_matrix_prediction[:, PowerFlowFeatureIds.voltage_magnitude]
    angles_rad = PQVA_matrix_prediction[:, PowerFlowFeatureIds.voltage_angle]

    active_errors = active_power_errors(edge_index=batch.edge_index,
                                        active_powers=active_powers,
                                        voltages=voltage_magnitudes,
                                        angles_rad=angles_rad,
                                        conductances=batch.conductances_pu,
                                        susceptances=batch.susceptances_pu)

    reactive_errors = reactive_power_errors(edge_index=batch.edge_index,
                                            reactive_powers=reactive_powers,
                                            voltages=voltage_magnitudes,
                                            angles_rad=angles_rad,
                                            conductances=batch.conductances_pu,
                                            susceptances=batch.susceptances_pu)

    relative_active_power_errors = torch.abs(relative_values(active_errors, active_powers))
    relative_reactive_power_errors = torch.abs(relative_values(reactive_errors, reactive_powers))

    return relative_active_power_errors, relative_reactive_power_errors
