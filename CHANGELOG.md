# Change log

## 0.0.1 (17/06/2023)

First release

## 0.0.5 (19/06/2023)

Add README.md as long description in PyPi.

## 0.0.6 (20/06/2023)

* Configure setup.py such that `pip install mlpf` works.
* Separate torch and numpy in data conversion

## 0.0.7(24/06/2023)

* Refactor power flow errors in such a way that the calculation is now
  separate for active and reactive power decoupling them
* Refactor TorchMetrics custom metrics to return only one metric