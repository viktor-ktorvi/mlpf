import torch

from torch import Tensor
from torch_geometric.data import Data
from torchmetrics import Metric

from mlpf.enumerations.optimal_power_flow_ids import OptimalPowerFlowFeatureIds
from mlpf.enumerations.power_flow_ids import PowerFlowFeatureIds
from mlpf.loss.torch.costs import polynomial_costs
from mlpf.loss.torch.metrics.utils import incorporate_predictions


def calculate_cost(PQVA_matrix_prediction: Tensor, data: Data):
    """
    Extract the active powers generation, the cost coefficients and baseMVA and use them to calculate the polynomial cost function value.

    :param PQVA_matrix_prediction:
    :param data:
    :return:
    """
    active_powers = PQVA_matrix_prediction[:, PowerFlowFeatureIds.active_power]
    active_power_demands = data.opf_features_matrix[:, OptimalPowerFlowFeatureIds.active_power_demands]

    active_powers_generation = (active_powers + active_power_demands) * data.baseMVA[data.batch]

    cost_coefficients = data.opf_features_matrix[:, OptimalPowerFlowFeatureIds.cost_coefficients_start:]

    return torch.sum(polynomial_costs(active_powers_generation, cost_coefficients))


class MeanActivePowerCost(Metric):
    """
    A TorchMetric class for calculating the average active power cost in a grid.

    TODO Latex
    """

    def __init__(self):
        super(MeanActivePowerCost, self).__init__()

        self.add_state("PQVA_matrix_prediction", default=torch.tensor(0.0))
        self.add_state("cost", default=torch.tensor(0.0))
        self.add_state("grid_count", default=torch.tensor(0))

    def update(self, power_flow_predictions: Tensor, batch: Data):
        self.PQVA_matrix_prediction = incorporate_predictions(power_flow_predictions, batch)

        self.cost += calculate_cost(self.PQVA_matrix_prediction, batch)

        self.grid_count += len(batch.target_cost)

    def compute(self) -> Tensor:
        return self.cost / self.grid_count

    @property
    def unit(self) -> str:
        return "$/h"


class MeanRelativeActivePowerCost(Metric):
    """
    A TorchMetric class for calculating the average ratio between the active power cost in a grid solved by the model and solved by an OPF solver.

    TODO Latex
    """

    def __init__(self):
        super(MeanRelativeActivePowerCost, self).__init__()

        self.add_state("PQVA_matrix_prediction", default=torch.tensor(0.0))
        self.add_state("cost", default=torch.tensor(0.0))
        self.add_state("target_cost", default=torch.tensor(0.0))

    def update(self, power_flow_predictions: Tensor, batch: Data):
        self.PQVA_matrix_prediction = incorporate_predictions(power_flow_predictions, batch)

        self.cost += calculate_cost(self.PQVA_matrix_prediction, batch)
        self.target_cost += torch.sum(batch.target_cost)

    def compute(self) -> Tensor:
        return self.cost / self.target_cost

    @property
    def unit(self) -> str:
        return "ratio"
