import copy

import pandapower as pp
import pandapower.networks as pn
import torch
from pypower.ppoption import ppoption
from pypower.runpf import runpf

from mlpf.data.conversion.torch.power_flow import ppc2power_flow_tensors
from mlpf.loss.torch.power_flow import power_flow_errors

net = pn.case118()

ppc = pp.converter.to_ppc(net, init="flat")

ppopt = ppoption(OUT_ALL=0, VERBOSE=0)
ppc, converged = runpf(copy.deepcopy(ppc), ppopt=ppopt)

# note: going from float64 to float32(the standard in torch) will increase the PF loss significantly
edge_index, active_powers_pu, reactive_powers_pu, voltages_pu, angles_rad, conductances_pu, susceptances_pu = ppc2power_flow_tensors(ppc, dtype=torch.float64)

active_power_losses_pu, reactive_power_losses_pu = power_flow_errors(
    edge_index,
    active_powers_pu,
    reactive_powers_pu,
    voltages_pu,
    angles_rad,
    conductances_pu,
    susceptances_pu
)

print(f"Total P loss = {torch.sum(active_power_losses_pu):.3e} p.u.")
print(f"Total Q loss = {torch.sum(reactive_power_losses_pu):.3e} p.u.")
