import torch

from torch import Tensor


def make_sparse_admittance_matrix(edge_index: Tensor,
                                  conductances: Tensor,
                                  susceptances: Tensor) -> Tensor:
    """
    Create a sparse admittance matrix from the edge index and corresponding conductance and susceptance values.

    :param edge_index: Edge list.
    :param conductances: Conductance values corresponding to the edge list.
    :param susceptances: Susceptance values corresponding to the edge list.
    :return: Sparse admittance matrix.
    """
    admittance_tensor = conductances + 1j * susceptances

    num_nodes = len(torch.unique(edge_index))
    shape = (num_nodes, num_nodes)

    return torch.sparse.FloatTensor(edge_index, admittance_tensor, torch.Size(shape))
