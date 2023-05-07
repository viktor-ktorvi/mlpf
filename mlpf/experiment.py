import torch

import pandapower as pp
import pandapower.networks as pn

from mlpf.data.utils.values_from_ppc import extract_values
from mlpf.loss.power_flow import power_flow_errors_pu, scalarize

if __name__ == "__main__":
    net = pn.create_kerber_dorfnetz()
    pp.runpp(net)

    ppc = net._ppc

    edge_index, active_powers_pu, reactive_powers_pu, voltages_pu, angles_deg, conductances_pu, susceptances_pu, baseMVA, basekV = extract_values(ppc, torch.float64)

    active_power_losses_pu, reactive_power_losses_pu = power_flow_errors_pu(edge_index,
                                                                            active_powers_pu,
                                                                            reactive_powers_pu,
                                                                            voltages_pu,
                                                                            angles_deg,
                                                                            conductances_pu,
                                                                            susceptances_pu,
                                                                            baseMVA,
                                                                            basekV)

    loss = scalarize(active_power_losses_pu, reactive_power_losses_pu)
    debug_var = None
