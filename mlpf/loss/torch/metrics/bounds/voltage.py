import torch

from torch import Tensor
from torch_geometric.data import Data
from torchmetrics import Metric

from mlpf.enumerations.optimal_power_flow_ids import OptimalPowerFlowFeatureIds
from mlpf.enumerations.power_flow_ids import PowerFlowFeatureIds
from mlpf.loss.torch.bound_errors import lower_bound_errors, upper_bound_errors
from mlpf.loss.torch.metrics.utils import incorporate_predictions


class MeanUpperVoltageError(Metric):
    """
    A TorchMetric class for calculating the average upper voltage bound error per node.

    TODO Latex
    """

    def __init__(self):
        super(MeanUpperVoltageError, self).__init__()

        self.add_state("PQVA_matrix_prediction", default=torch.tensor(0.0))
        self.add_state("upper_voltage_error_sum", default=torch.tensor(0.0))
        self.add_state("node_count", default=torch.tensor(0))

    def update(self, power_flow_predictions: Tensor, batch: Data):
        self.PQVA_matrix_prediction = incorporate_predictions(power_flow_predictions, batch)

        self.upper_voltage_error_sum += torch.sum(
            upper_bound_errors(
                value=self.PQVA_matrix_prediction[:, PowerFlowFeatureIds.voltage_magnitude],
                value_max=batch.opf_features_matrix[:, OptimalPowerFlowFeatureIds.voltages_max]
            )
        )

        self.node_count += self.PQVA_matrix_prediction.shape[0]

    def compute(self) -> Tensor:
        return self.upper_voltage_error_sum / self.node_count

    @property
    def unit(self) -> str:
        return "p.u."


class MeanLowerVoltageError(Metric):
    """
    A TorchMetric class for calculating the average lower voltage bound error per node.

    TODO Latex
    """

    def __init__(self):
        super(MeanLowerVoltageError, self).__init__()

        self.add_state("PQVA_matrix_prediction", default=torch.tensor(0.0))
        self.add_state("lower_voltage_error_sum", default=torch.tensor(0.0))
        self.add_state("node_count", default=torch.tensor(0))

    def update(self, power_flow_predictions: Tensor, batch: Data):
        self.PQVA_matrix_prediction = incorporate_predictions(power_flow_predictions, batch)

        self.lower_voltage_error_sum += torch.sum(
            lower_bound_errors(
                value=self.PQVA_matrix_prediction[:, PowerFlowFeatureIds.voltage_magnitude],
                value_min=batch.opf_features_matrix[:, OptimalPowerFlowFeatureIds.voltages_min]
            )
        )

        self.node_count += self.PQVA_matrix_prediction.shape[0]

    def compute(self) -> Tensor:
        return self.lower_voltage_error_sum / self.node_count

    @property
    def unit(self) -> str:
        return "p.u."
