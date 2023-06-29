import numpy as np

from numpy import ndarray
from typing import Final, List

from mlpf.enumerations.bus_type import BusTypeIds
from mlpf.enumerations.power_flow_ids import PowerFlowFeatureIds


class BusTypeMasks:
    """
    A class with static fields defining bus type masks in the (active power, reactive power, voltage magnitude, voltage angle) format.
    A True value specifies that the value is known.
    """
    Slack: Final[List[bool]] = [False, False, True, True]
    PQ: Final[List[bool]] = [True, True, False, False]
    PV: Final[List[bool]] = [True, False, True, False]


def create_power_flow_feature_mask(bus_types: ndarray) -> ndarray:
    """
    Create a feature mask depending on the node types in the (active power, reactive power, voltage magnitude, voltage angle) format.

    :param bus_types: The bus types column from the bus table of the pypower case format.
    :return: Boolean feature mask.
    """
    feature_mask = np.zeros((len(bus_types), len(PowerFlowFeatureIds)), dtype=bool)

    slack_indices = (bus_types == BusTypeIds.Slack).nonzero()[0]
    pq_indices = (bus_types == BusTypeIds.PQ).nonzero()[0]
    pv_indices = (bus_types == BusTypeIds.PV).nonzero()[0]

    feature_mask[slack_indices] = np.array(BusTypeMasks.Slack)
    feature_mask[pq_indices] = np.array(BusTypeMasks.PQ)
    feature_mask[pv_indices] = np.array(BusTypeMasks.PV)

    return feature_mask
