import copy
from typing import Dict

from pypower.ppoption import ppoption
from pypower.runpf import runpf


def ppc_runpf(ppc: Dict) -> Dict:
    """
    Run a power flow on a PPC.

    :param ppc: PyPower case object.
    :return: PF solved PPC.
    """
    ppopt = ppoption(OUT_ALL=0, VERBOSE=0)
    ppc, converged = runpf(copy.deepcopy(ppc), ppopt=ppopt)

    assert converged, "Power flow hasn't converged."

    return ppc
