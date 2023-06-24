import torch
from torch import Tensor
from torch_geometric.data import Data
from torchmetrics import Metric

from mlpf.enumerations.power_flow_ids import PowerFlowFeatureIds
from mlpf.loss.relative_values import relative_values
from mlpf.loss.torch.metrics.utils import incorporate_predictions
from mlpf.loss.torch.power_flow import reactive_power_errors


class MeanReactivePowerError(Metric):
    """
    A TorchMetric class for calculating the average absolute reactive power flow error.

    TODO Latex
    """

    def __init__(self):
        super(MeanReactivePowerError, self).__init__()

        self.add_state("reactive_errors", default=torch.tensor(0.0))
        self.add_state("reactive_error_sum", default=torch.tensor(0.0))
        self.add_state("PQVA_matrix_prediction", default=torch.tensor(0.0))

        self.add_state("node_count", default=torch.tensor(0))

    def update(self, preds_pf: Tensor, batch: Data):
        self.PQVA_matrix_prediction = incorporate_predictions(preds_pf, batch)

        self.reactive_errors = reactive_power_errors(edge_index=batch.edge_index,
                                                     reactive_powers=self.PQVA_matrix_prediction[:, PowerFlowFeatureIds.reactive_power],
                                                     voltages=self.PQVA_matrix_prediction[:, PowerFlowFeatureIds.voltage_magnitude],
                                                     angles_rad=self.PQVA_matrix_prediction[:, PowerFlowFeatureIds.voltage_angle],
                                                     conductances=batch.conductances_pu,
                                                     susceptances=batch.susceptances_pu)

        self.reactive_error_sum += torch.sum(torch.abs(self.reactive_errors))
        self.node_count += self.reactive_errors.shape[0]

    def compute(self) -> Tensor:
        return self.reactive_error_sum / self.node_count


class MeanRelativeReactivePowerError(Metric):
    """
    A TorchMetric class for calculating the average relative absolute reactive power flow error.

    TODO Latex
    """

    def __init__(self):
        super(MeanRelativeReactivePowerError, self).__init__()

        self.add_state("reactive_errors", default=torch.tensor(0.0))
        self.add_state("relative_reactive_error_sum", default=torch.tensor(0.0))
        self.add_state("PQVA_matrix_prediction", default=torch.tensor(0.0))

        self.add_state("node_count", default=torch.tensor(0))

    def update(self, preds_pf: Tensor, batch: Data):
        self.PQVA_matrix_prediction = incorporate_predictions(preds_pf, batch)

        self.reactive_errors = reactive_power_errors(edge_index=batch.edge_index,
                                                     reactive_powers=self.PQVA_matrix_prediction[:, PowerFlowFeatureIds.reactive_power],
                                                     voltages=self.PQVA_matrix_prediction[:, PowerFlowFeatureIds.voltage_magnitude],
                                                     angles_rad=self.PQVA_matrix_prediction[:, PowerFlowFeatureIds.voltage_angle],
                                                     conductances=batch.conductances_pu,
                                                     susceptances=batch.susceptances_pu)

        self.relative_reactive_error_sum += torch.sum(torch.abs(relative_values(self.reactive_errors, self.PQVA_matrix_prediction[:, PowerFlowFeatureIds.reactive_power])))
        self.node_count += self.reactive_errors.shape[0]

    def compute(self) -> Tensor:
        return self.relative_reactive_error_sum / self.node_count
