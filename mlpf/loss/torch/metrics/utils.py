import torch

from torch import Tensor
from torch_geometric.data import Data


def incorporate_predictions(predictions: Tensor, batch: Data):
    """
    Fill the PQVA matrix with the calculated predictions of the problem variables.

    :param predictions: Variable predictions.
    :param batch: PyG Data batch containing all the power flow information.
    :return: Solved PQVA matrix.
    """
    PQVA_matrix_prediction = torch.zeros_like(batch.PQVA_matrix)

    PQVA_matrix_prediction[batch.feature_mask] = batch.PQVA_matrix[batch.feature_mask]
    PQVA_matrix_prediction[~batch.feature_mask] = predictions.flatten()

    return PQVA_matrix_prediction
