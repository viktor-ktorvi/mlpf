import numpy as np

from numpy import ndarray
from typing import Union

from mlpf.data.data.optimal_power_flow import OptimalPowerFlowData
from mlpf.data.data.power_flow import PowerFlowData


def incorporate_predictions(predictions: ndarray, data: Union[PowerFlowData, OptimalPowerFlowData]):
    """
    Merge the known fields in the PQVA matrix with the predictions fo the unknown fields.

    :param predictions: Predicted unknown fields.
    :param data: OPF data object.
    :return: PQVA matrix prediction.
    """
    PQVA_matrix_prediction = np.zeros_like(data.PQVA_matrix)
    PQVA_matrix_prediction[data.feature_mask] = data.PQVA_matrix[data.feature_mask]
    PQVA_matrix_prediction[~data.feature_mask] = predictions.flatten()

    return PQVA_matrix_prediction
