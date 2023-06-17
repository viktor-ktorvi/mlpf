from typing import Dict

import torch

from torch import Tensor
from torch_geometric.data import Data
from torchmetrics import Metric

from mlpf.data.data.torch.power_flow import get_relative_power_flow_errors


class RelativePowerFlowError(Metric):
    def __init__(self,
                 active_str: str = "relative_active_power_errors",
                 reactive_str: str = "relative_reactive_power_errors"):
        """
        Initialize the metric states.
        :param active_str: Output dictionary key for the relative active power error.
        :param reactive_str: Output dictionary key for the relative reactive power error.
        """
        super(RelativePowerFlowError, self).__init__()

        self.active_str = active_str
        self.reactive_str = reactive_str

        self.add_state("relative_active_power_errors", default=[])
        self.add_state("relative_reactive_power_errors", default=[])

    def update(self, preds_pf: Tensor, batch: Data):
        """
        Calculate the relative power flow errors and append them to the states.

        :param preds_pf: Predictions(physical values).
        :param batch: Corresponding batch Data object.
        :return:
        """
        relative_active_power_errors, relative_reactive_power_errors = get_relative_power_flow_errors(preds_pf.detach().cpu(), batch.detach().cpu())

        self.relative_active_power_errors.append(relative_active_power_errors)
        self.relative_reactive_power_errors.append(relative_reactive_power_errors)

    def compute(self) -> Dict:
        """
        Calculate the mean and median relative errors across all the elements.

        :return: A dictionary with the final metric values.
        """
        assert len(self.relative_active_power_errors) == len(self.relative_reactive_power_errors)

        relative_active_power_errors = self.relative_active_power_errors[0]
        relative_reactive_power_errors = self.relative_reactive_power_errors[0]

        for i in range(1, len(self.relative_active_power_errors)):
            relative_active_power_errors = torch.cat((relative_active_power_errors, self.relative_active_power_errors[i]))
            relative_reactive_power_errors = torch.cat((relative_reactive_power_errors, self.relative_reactive_power_errors[i]))

        return {
            self.active_str + " mean": relative_active_power_errors.mean(),
            self.reactive_str + " mean": relative_reactive_power_errors.mean(),
            self.active_str + " median": relative_active_power_errors.median(),
            self.reactive_str + " median": relative_reactive_power_errors.median()
        }
