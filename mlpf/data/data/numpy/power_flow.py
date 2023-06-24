import copy

import numpy as np

from numpy import ndarray
from types import SimpleNamespace
from typing import Tuple

from mlpf.data.conversion.numpy.power_flow import ppc2power_flow_arrays
from mlpf.data.utils.masks import create_feature_mask
from mlpf.enumerations.bus_table import BusTableIds
from mlpf.enumerations.power_flow_ids import PowerFlowFeatureIds
from mlpf.loss.numpy.power_flow import active_power_errors, reactive_power_errors
from mlpf.loss.relative_values import relative_values
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
    feature_mask = create_feature_mask(ppc["bus"][:, BusTableIds.bus_type])

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


def get_relative_power_flow_errors(predictions: ndarray, data: SimpleNamespace) -> Tuple[ndarray, ...]:
    """
    Take model predictions and merge them with the corresponding data batch.
    Calculate the relative power flow errors for the given predictions.

    :param predictions: Power flow unknown variable predictions.
    :param data: Data object from which the predictions came from.
    :return: (relative active power errors, relative reactive power errors)
    """
    PQVA_matrix_prediction = copy.deepcopy(data.PQVA_matrix)
    PQVA_matrix_prediction[~data.feature_mask] = predictions.flatten()

    active_powers = PQVA_matrix_prediction[:, PowerFlowFeatureIds.active_power]
    reactive_powers = PQVA_matrix_prediction[:, PowerFlowFeatureIds.reactive_power]

    voltage_magnitudes = PQVA_matrix_prediction[:, PowerFlowFeatureIds.voltage_magnitude]
    angles_rad = PQVA_matrix_prediction[:, PowerFlowFeatureIds.voltage_angle]

    active_errors = active_power_errors(edge_index=data.edge_index,
                                        active_powers=active_powers,
                                        voltages=voltage_magnitudes,
                                        angles_rad=angles_rad,
                                        conductances=data.conductances_pu,
                                        susceptances=data.susceptances_pu)

    reactive_errors = reactive_power_errors(edge_index=data.edge_index,
                                            reactive_powers=reactive_powers,
                                            voltages=voltage_magnitudes,
                                            angles_rad=angles_rad,
                                            conductances=data.conductances_pu,
                                            susceptances=data.susceptances_pu)

    relative_active_power_errors = np.abs(relative_values(active_errors, active_powers))
    relative_reactive_power_errors = np.abs(relative_values(reactive_errors, reactive_powers))

    return relative_active_power_errors, relative_reactive_power_errors
