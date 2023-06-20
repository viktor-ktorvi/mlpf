from typing import List, Dict

import pandapower as pp
from pandapower import pandapowerNet


def pandapower2ppc_list(pandapower_networks: List[pandapowerNet]) -> List[Dict]:
    """
    Convert a list of pandapower networks to a list of pypower case files.

    :param pandapower_networks:
    :return:
    """
    ppc_list = []
    for net in pandapower_networks:
        pp.runpp(net, numba=False)
        ppc_list.append(net._ppc)

    return ppc_list
