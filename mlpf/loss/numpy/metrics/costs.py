import numpy as np

from numpy import ndarray

from mlpf.data.data.optimal_power_flow import OptimalPowerFlowData
from mlpf.enumerations.optimal_power_flow_ids import OptimalPowerFlowFeatureIds
from mlpf.enumerations.power_flow_ids import PowerFlowFeatureIds
from mlpf.loss.numpy.costs import polynomial_costs
from mlpf.loss.numpy.metrics.metrics import BaseMetric
from mlpf.loss.numpy.metrics.utils import incorporate_predictions


def calculate_cost(PQVA_matrix_prediction: ndarray, data: OptimalPowerFlowData):
    """
    Extract the active powers generation, the cost coefficients and baseMVA and use them to calculate the polynomial cost function value.

    :param PQVA_matrix_prediction:
    :param data:
    :return:
    """
    active_powers = PQVA_matrix_prediction[:, PowerFlowFeatureIds.active_power]
    active_power_demands = data.opf_features_matrix[:, OptimalPowerFlowFeatureIds.active_power_demands]

    active_powers_generation = (active_powers + active_power_demands) * data.baseMVA

    cost_coefficients = data.opf_features_matrix[:, OptimalPowerFlowFeatureIds.cost_coefficients_start:]

    return np.sum(polynomial_costs(active_powers_generation, cost_coefficients))


class ActivePowerCost(BaseMetric):
    """
    Active power cost function metric per grid.

    TODO Latex
    """

    def __init__(self):
        super(ActivePowerCost, self).__init__()
        self.costs = []

    def update(self, predictions: ndarray, data: OptimalPowerFlowData):
        PQVA_matrix_prediction = incorporate_predictions(predictions, data)

        self.costs.append(calculate_cost(PQVA_matrix_prediction, data))

    def compute(self) -> ndarray:
        return np.array(self.costs)

    @property
    def unit(self) -> str:
        return "$/hr"


class RelativeActivePowerCost(BaseMetric):
    """
    Relative active power cost function metric per grid i.e. the ratio of the cost achieved by the model prediction and
     the cost achieved by an OPF solver (provided in the dataset).

    TODO Latex
    """

    def __init__(self):
        super(RelativeActivePowerCost, self).__init__()
        self.relative_active_power_costs = []

    def update(self, predictions, data):
        PQVA_matrix_prediction = incorporate_predictions(predictions, data)
        cost = calculate_cost(PQVA_matrix_prediction, data)

        self.relative_active_power_costs.append(cost / data.target_cost)

    def compute(self) -> ndarray:
        return np.array(self.relative_active_power_costs)

    @property
    def unit(self) -> str:
        return "ratio"
