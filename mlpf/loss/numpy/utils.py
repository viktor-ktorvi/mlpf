import numpy as np
from numpy import ndarray

from scipy.sparse import coo_matrix


def make_sparse_admittance_matrix(edge_index: ndarray,
                                  conductances: ndarray,
                                  susceptances: ndarray) -> coo_matrix:
    """
    Create a sparse admittance matrix from the edge index and corresponding conductance and susceptance values.

    :param edge_index: Edge list.
    :param conductances: Conductance values corresponding to the edge list.
    :param susceptances: Susceptance values corresponding to the edge list.
    :return: Sparse admittance matrix.
    """
    admittance_array = conductances + 1j * susceptances

    num_nodes = len(np.unique(edge_index))
    shape = (num_nodes, num_nodes)

    return coo_matrix((admittance_array, (edge_index[0], edge_index[1])), shape=shape)
