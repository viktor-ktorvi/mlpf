import numpy as np

from numpy import ndarray
from types import SimpleNamespace

from mlpf.enumerations.optimal_power_flow_ids import OptimalPowerFlowFeatureIds
from mlpf.enumerations.power_flow_ids import PowerFlowFeatureIds
from mlpf.loss.numpy.bound_errors import upper_bound_errors, lower_bound_errors
from mlpf.loss.numpy.metrics.metrics import BaseMetric
from mlpf.loss.numpy.metrics.utils import incorporate_predictions


class UpperVoltageError(BaseMetric):
    """
    Upper voltage magnitude bound error/violation metric calculated for every node in the grid/dataset.

    TODO Latex
    """

    def __init__(self):
        super(UpperVoltageError, self).__init__()
        self.upper_voltage_errors = []

    def update(self, predictions: ndarray, data: SimpleNamespace):
        PQVA_matrix_prediction = incorporate_predictions(predictions, data)

        self.upper_voltage_errors.append(
            upper_bound_errors(
                value=PQVA_matrix_prediction[:, PowerFlowFeatureIds.voltage_magnitude],
                value_max=data.opf_features_matrix[:, OptimalPowerFlowFeatureIds.voltages_max]
            )
        )

    def compute(self) -> ndarray:
        return np.array(self.upper_voltage_errors)

    @property
    def unit(self) -> str:
        return "p.u."


class LowerVoltageError(BaseMetric):
    """
    Lower voltage magnitude bound error/violation metric calculated for every node in the grid/dataset.

    TODO Latex
    """

    def __init__(self):
        super(LowerVoltageError, self).__init__()
        self.lower_voltage_errors = []

    def update(self, predictions: ndarray, data: SimpleNamespace):
        PQVA_matrix_prediction = incorporate_predictions(predictions, data)

        self.lower_voltage_errors.append(
            lower_bound_errors(
                value=PQVA_matrix_prediction[:, PowerFlowFeatureIds.voltage_magnitude],
                value_min=data.opf_features_matrix[:, OptimalPowerFlowFeatureIds.voltages_min]
            )
        )

    def compute(self) -> ndarray:
        return np.array(self.lower_voltage_errors)

    @property
    def unit(self) -> str:
        return "p.u."
