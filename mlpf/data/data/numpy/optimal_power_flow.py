import numpy as np

from pypower.ppoption import ppoption
from pypower.runopf import runopf
from types import SimpleNamespace

from mlpf.data.conversion.numpy.optimal_power_flow import ppc2optimal_power_flow_arrays
from mlpf.data.conversion.numpy.power_flow import ppc2power_flow_arrays
from mlpf.data.masks.optimal_power_flow import create_feature_mask
from mlpf.enumerations.bus_table import BusTableIds


def optimal_power_flow_data(ppc: dict, solve: bool = False, dtype: np.dtype = np.float64):
    """
    Extract all the data of the optimal power flow problem into a namespace object. The ppc should have a solved optimal power flow;
    otherwise if solve is set to True, and optimal power flow will be run.

    :param ppc: PYPOWER case object.
    :param solve: To run an OPF solver or not.
    :param dtype: Data type into which to cast the data.
    :return: Namespace object.
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

    feature_mask = create_feature_mask(ppc["bus"][:, BusTableIds.bus_type])  # PQVA mask
    feature_vector = np.hstack((PQVA_matrix, opf_features_matrix))[np.hstack((feature_mask, np.ones(shape=opf_features_matrix.shape, dtype=bool)))]

    edge_attributes = np.vstack((conductances_pu, susceptances_pu))

    return SimpleNamespace(
        edge_index=edge_index,
        edge_attr=edge_attributes,
        PQVA_matrix=PQVA_matrix,
        opf_features_matrix=opf_features_matrix,
        feature_mask=feature_mask,
        conductances_pu=conductances_pu,
        susceptances_pu=susceptances_pu,
        feature_vector=feature_vector,
        target_vector=PQVA_matrix[~feature_mask],
        target_cost=ppc["f"],
        baseMVA=ppc["baseMVA"]
    )
