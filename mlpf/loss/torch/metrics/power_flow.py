import torch
from torch import Tensor
from torch_geometric.data import Data
from torchmetrics import Metric

from mlpf.enumerations.power_flow_ids import PowerFlowFeatureIds
from mlpf.loss.relative_values import relative_values
from mlpf.loss.torch.power_flow import active_power_errors


def incorporate_predictions(predictions: Tensor, batch: Data):
    PQVA_matrix_prediction = torch.zeros_like(batch.PQVA_matrix)

    PQVA_matrix_prediction[batch.feature_mask] = batch.PQVA_matrix[batch.feature_mask]
    PQVA_matrix_prediction[~batch.feature_mask] = predictions.flatten()

    return PQVA_matrix_prediction


class MeanActivePowerError(Metric):
    def __init__(self):
        super(MeanActivePowerError, self).__init__()

        self.add_state("active_errors", default=torch.tensor(0.0))
        self.add_state("active_error_sum", default=torch.tensor(0.0))
        self.add_state("PQVA_matrix_prediction", default=torch.tensor(0.0))

        self.add_state("node_count", default=torch.tensor(0))

    def update(self, preds_pf: Tensor, batch: Data):
        self.PQVA_matrix_prediction = incorporate_predictions(preds_pf, batch)

        self.active_errors = active_power_errors(edge_index=batch.edge_index,
                                                 active_powers=self.PQVA_matrix_prediction[:, PowerFlowFeatureIds.active_power],
                                                 voltages=self.PQVA_matrix_prediction[:, PowerFlowFeatureIds.voltage_magnitude],
                                                 angles_rad=self.PQVA_matrix_prediction[:, PowerFlowFeatureIds.voltage_angle],
                                                 conductances=batch.conductances_pu,
                                                 susceptances=batch.susceptances_pu)

        self.active_error_sum += torch.sum(torch.abs(self.active_errors))
        self.node_count += self.active_errors.shape[0]

    def compute(self) -> Tensor:
        return self.active_error_sum / self.node_count


class MeanRelativeActivePowerError(Metric):
    def __init__(self):
        super(MeanRelativeActivePowerError, self).__init__()

        self.add_state("active_errors", default=torch.tensor(0.0))
        self.add_state("relative_active_error_sum", default=torch.tensor(0.0))
        self.add_state("PQVA_matrix_prediction", default=torch.tensor(0.0))

        self.add_state("node_count", default=torch.tensor(0))

    def update(self, preds_pf: Tensor, batch: Data):
        self.PQVA_matrix_prediction = incorporate_predictions(preds_pf, batch)

        self.active_errors = active_power_errors(edge_index=batch.edge_index,
                                                 active_powers=self.PQVA_matrix_prediction[:, PowerFlowFeatureIds.active_power],
                                                 voltages=self.PQVA_matrix_prediction[:, PowerFlowFeatureIds.voltage_magnitude],
                                                 angles_rad=self.PQVA_matrix_prediction[:, PowerFlowFeatureIds.voltage_angle],
                                                 conductances=batch.conductances_pu,
                                                 susceptances=batch.susceptances_pu)

        self.relative_active_error_sum += torch.sum(torch.abs(relative_values(self.active_errors, self.PQVA_matrix_prediction[:, PowerFlowFeatureIds.active_power])))
        self.node_count += self.active_errors.shape[0]

    def compute(self) -> Tensor:
        return self.relative_active_error_sum / self.node_count
