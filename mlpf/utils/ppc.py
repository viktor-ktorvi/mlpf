import copy
from typing import Dict

from pypower.ppoption import ppoption
from pypower.runopf import runopf
from pypower.runpf import runpf


def ppc_runpf(ppc: Dict) -> Dict:
    """
    Run a power flow on a PPC.

    :param ppc: PyPower case object.
    :return: PF solved PPC.
    """
    ppopt = ppoption(OUT_ALL=0, VERBOSE=0)
    pf_ppc, converged = runpf(copy.deepcopy(ppc), ppopt=ppopt)

    assert converged, "Power flow hasn't converged."

    return pf_ppc


def ppc_runopf(ppc: Dict) -> Dict:
    """
    Run an optimal power flow on a PPC.

    :param ppc: PyPower case object.
    :return: OPF solved PPC.
    """
    ppopt = ppoption(OUT_ALL=0, VERBOSE=0)
    opf_ppc = runopf(ppc, ppopt=ppopt)

    assert opf_ppc['success'], "Optimal power flow hasn't converged."

    return opf_ppc
