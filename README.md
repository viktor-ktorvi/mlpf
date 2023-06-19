<p align="center">
<img src="https://github.com/viktor-ktorvi/mlpf/assets/69254199/333dfd18-7c60-4874-a89b-92eecf32ac96?raw=True" width="650">
</p>



__MLPF__ is a python library for (optimal) power flow calculations with machine learning.
It offers:

* efficient loss functions compatible with both _PyTorch_ and _scikit-learn_!
* utilities such as data structures and loading pipelines that make it easy to go from
  _pandapower_ nets or _PYPOWER_ case files to arrays and tensors in just one line of code!
* visualization and description tools to take a quick look at your data

Contributions welcome!

## Installation

```commandline
pip install mlpf
```

## Usage

1. Load/create data and turn it into the [PYPOWER case format](https://rwl.github.io/PYPOWER/api/pypower.caseformat-module.html)
2. From then on we provide functionality to express the data as numpy arrays or torch tensors.
3. Feed that data into your ML (scikit-learn or torch) models and use our tried and tested loss functions to train, validate or monitor your model development.

```python
import copy

import pandapower as pp
import pandapower.networks as pn

from pypower.ppoption import ppoption
from pypower.runpf import runpf

net = pn.case118()

ppc = pp.converter.to_ppc(net, init="flat")

ppopt = ppoption(OUT_ALL=0, VERBOSE=0)
ppc, converged = runpf(copy.deepcopy(ppc), ppopt=ppopt)
```

### Loss

#### numpy / scikit-learn

```python
from mlpf.data.utils.conversion import ppc2power_flow_arrays
from mlpf.loss.numpy.power_flow import power_flow_errors

edge_index, active_powers_pu, reactive_powers_pu, voltages_pu, angles_rad, conductances_pu, susceptances_pu = ppc2power_flow_arrays(ppc)

active_power_losses_pu, reactive_power_losses_pu = power_flow_errors(
    edge_index,
    active_powers_pu,
    reactive_powers_pu,
    voltages_pu, angles_rad,
    conductances_pu,
    susceptances_pu
)
```

#### torch

```python
from mlpf.data.utils.conversion import ppc2power_flow_tensors
from mlpf.loss.torch.power_flow import power_flow_errors

# note: going from float64(standard in PYPOWER) to float32(standard in torch) will increase the PF loss significantly
edge_index, active_powers_pu, reactive_powers_pu, voltages_pu, angles_rad, conductances_pu, susceptances_pu = ppc2power_flow_tensors(ppc, dtype=torch.float64)

active_power_losses_pu, reactive_power_losses_pu = power_flow_errors(
    edge_index,
    active_powers_pu,
    reactive_powers_pu,
    voltages_pu, angles_rad,
    conductances_pu,
    susceptances_pu
)
```

### Data loading

- [ ] TODO

### Indepth examples

### Development

```
git clone https://github.com/viktor-ktorvi/mlpf.git
cd mlpf

conda env create -f environment.yml
conda activate mlpfenv
```
