import numpy as np

from numpy import ndarray

from mlpf.data.data.numpy.optimal_power_flow import OptimalPowerFlowData
from mlpf.enumerations.optimal_power_flow_ids import OptimalPowerFlowFeatureIds
from mlpf.enumerations.power_flow_ids import PowerFlowFeatureIds
from mlpf.loss.numpy.bound_errors import upper_bound_errors, lower_bound_errors
from mlpf.loss.numpy.metrics.metrics import BaseMetric
from mlpf.loss.numpy.metrics.utils import incorporate_predictions


class UpperReactivePowerError(BaseMetric):
    """
    Upper reactive power bound error/violation metric calculated for every node in the grid/dataset.

    TODO Latex
    """

    def __init__(self):
        super(UpperReactivePowerError, self).__init__()
        self.upper_reactive_power_errors = []

    def update(self, predictions: ndarray, data: OptimalPowerFlowData):
        PQVA_matrix_prediction = incorporate_predictions(predictions, data)

        self.upper_reactive_power_errors.append(
            upper_bound_errors(
                value=PQVA_matrix_prediction[:, PowerFlowFeatureIds.reactive_power],
                value_max=data.opf_features_matrix[:, OptimalPowerFlowFeatureIds.reactive_powers_max]
            )
        )

    def compute(self) -> ndarray:
        return np.array(self.upper_reactive_power_errors)

    @property
    def unit(self) -> str:
        return "p.u."


class LowerReactivePowerError(BaseMetric):
    """
    Lower reactive power bound error/violation metric calculated for every node in the grid/dataset.

    TODO Latex
    """

    def __init__(self):
        super(LowerReactivePowerError, self).__init__()
        self.lower_reactive_power_errors = []

    def update(self, predictions: ndarray, data: OptimalPowerFlowData):
        PQVA_matrix_prediction = incorporate_predictions(predictions, data)

        self.lower_reactive_power_errors.append(
            lower_bound_errors(
                value=PQVA_matrix_prediction[:, PowerFlowFeatureIds.reactive_power],
                value_min=data.opf_features_matrix[:, OptimalPowerFlowFeatureIds.reactive_powers_min]
            )
        )

    def compute(self) -> ndarray:
        return np.array(self.lower_reactive_power_errors)

    @property
    def unit(self) -> str:
        return "p.u."
