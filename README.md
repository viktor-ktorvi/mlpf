<p align="center">
<img src="https://github.com/viktor-ktorvi/mlpf/assets/69254199/333dfd18-7c60-4874-a89b-92eecf32ac96?raw=True" width="650">
</p>



__MLPF__ is a python library for (optimal) power flow calculations with machine learning.
It offers:

* efficient loss functions compatible with both _PyTorch_ and _scikit-learn_
* utilities such as data structures and loading pipelines that make it easy to go from
  _pandapower_ nets or _PYPOWER_ case files to arrays and tensors in just one line of code
* visualization and description tools to take a quick look at your data

Contributions welcome!

## Installation

```commandline
pip install mlpf
```

The previous command will install all the dependencies for working with numpy and scikit-learn. It will **not**, however, install all the dependencies needed for
working in torch. To use the torch functionalities, please
install [PyTorch](https://pytorch.org/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) with its dependencies(torch-scatter etc.)
and optionally [TorchMetrics](https://torchmetrics.readthedocs.io/en/stable/).

## Usage

1. Load/create data and turn it into the [PYPOWER case format](https://rwl.github.io/PYPOWER/api/pypower.caseformat-module.html)
2. From then on we provide functionality to express the data as numpy arrays or torch tensors.
3. Feed that data into your ML (scikit-learn or torch) models and use our tried and tested loss functions to train, validate or monitor your model development.

```python
import pandapower as pp
import pandapower.networks as pn

from pypower.ppoption import ppoption
from pypower.runpf import runpf

net = pn.case118()

ppc = pp.converter.to_ppc(net, init="flat")

ppopt = ppoption(OUT_ALL=0, VERBOSE=0)
ppc, converged = runpf(ppc, ppopt=ppopt)
```

### Loss

#### numpy / scikit-learn

```python
import numpy as np

from mlpf.data.conversion.numpy.power_flow import ppc2power_flow_arrays
from mlpf.loss.numpy.power_flow import active_power_errors, reactive_power_errors

edge_index, active_powers_pu, reactive_powers_pu, voltages_pu, angles_rad, conductances_pu, susceptances_pu = ppc2power_flow_arrays(ppc)

active_errors = active_power_errors(edge_index, active_powers_pu, voltages_pu, angles_rad, conductances_pu, susceptances_pu)
reactive_errors = reactive_power_errors(edge_index, reactive_powers_pu, voltages_pu, angles_rad, conductances_pu, susceptances_pu)

print(f"Total P loss = {np.sum(active_errors):.3e} p.u.")
print(f"Total Q loss = {np.sum(reactive_errors):.3e} p.u.")
```

#### torch

```python
import torch

from mlpf.data.conversion.torch.power_flow import ppc2power_flow_tensors
from mlpf.loss.torch.power_flow import active_power_errors, reactive_power_errors

# note: going from float64(standard in PYPOWER) to float32(standard in torch) will increase the PF loss significantly
edge_index, active_powers_pu, reactive_powers_pu, voltages_pu, angles_rad, conductances_pu, susceptances_pu = ppc2power_flow_tensors(ppc, dtype=torch.float64)

active_errors = active_power_errors(edge_index, active_powers_pu, voltages_pu, angles_rad, conductances_pu, susceptances_pu)
reactive_errors = reactive_power_errors(edge_index, reactive_powers_pu, voltages_pu, angles_rad, conductances_pu, susceptances_pu)

print(f"Total P loss = {torch.sum(active_errors):.3e} p.u.")
print(f"Total Q loss = {torch.sum(reactive_errors):.3e} p.u.")
```

### Data loading

- [ ] TODO

### Indepth examples

#### General

* [NumPy/scikit-learn loss](examples/sklearn/loss/from_arrays.py)
* [Torch loss](examples/torch/loss/from_arrays.py)

#### Supervised learning

##### Power flow

* [scikit-learn linear regression](examples/sklearn/supervised_power_flow/linear_regression.py)
* [torch MLP(multilayer perceptron)](examples/torch/supervised_power_flow/mlp.py)
* [torch GCN(graph convolutional network)](examples/torch/supervised_power_flow/gcn.py)

#### Unsupervised learning

##### Power flow

* [torch MLP(multilayer perceptron](examples/torch/unsupervised_power_flow/mlp.py)
* [torch GCN(graph convolutional network)](examples/torch/unsupervised_power_flow/gcn.py)

### Development

```
git clone https://github.com/viktor-ktorvi/mlpf.git
cd mlpf

conda env create -f environment.yml
conda activate mlpfenv
```
