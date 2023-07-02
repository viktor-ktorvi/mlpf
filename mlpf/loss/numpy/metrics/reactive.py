import numpy as np

from numpy import ndarray
from typing import Union

from mlpf.data.data.optimal_power_flow import OptimalPowerFlowData
from mlpf.data.data.power_flow import PowerFlowData
from mlpf.enumerations.power_flow_ids import PowerFlowFeatureIds
from mlpf.loss.numpy.metrics.metrics import BaseMetric
from mlpf.loss.numpy.metrics.utils import incorporate_predictions
from mlpf.loss.numpy.power_flow import reactive_power_errors
from mlpf.loss.relative_values import relative_values


class ReactivePowerError(BaseMetric):
    """
    Absolute reactive power error metric calculated for every node in the grid/dataset.

    TODO Latex
    """

    def __init__(self):
        super(ReactivePowerError, self).__init__()
        self.reactive_power_errors = []

    def update(self, predictions: ndarray, data: Union[PowerFlowData, OptimalPowerFlowData]):
        PQVA_matrix_prediction = incorporate_predictions(predictions, data)

        self.reactive_power_errors.append(
            np.abs(
                reactive_power_errors(
                    edge_index=data.edge_index,
                    reactive_powers=PQVA_matrix_prediction[:, PowerFlowFeatureIds.reactive_power],
                    voltages=PQVA_matrix_prediction[:, PowerFlowFeatureIds.voltage_magnitude],
                    angles_rad=PQVA_matrix_prediction[:, PowerFlowFeatureIds.voltage_angle],
                    conductances=data.conductances_pu,
                    susceptances=data.susceptances_pu
                )
            )
        )

    def compute(self) -> ndarray:
        return np.array(self.reactive_power_errors)

    @property
    def unit(self) -> str:
        return "p.u."


class RelativeReactivePowerError(BaseMetric):
    """
    Relative absolute reactive power error metric calculated for every node in the grid/dataset.

    TODO Latex
    """

    def __init__(self):
        super(RelativeReactivePowerError, self).__init__()
        self.relative_reactive_power_errors = []

    def update(self, predictions: ndarray, data: Union[PowerFlowData, OptimalPowerFlowData]):
        PQVA_matrix_prediction = incorporate_predictions(predictions, data)
        rel_errors = relative_values(
            numerator=reactive_power_errors(
                edge_index=data.edge_index,
                reactive_powers=PQVA_matrix_prediction[:, PowerFlowFeatureIds.reactive_power],
                voltages=PQVA_matrix_prediction[:, PowerFlowFeatureIds.voltage_magnitude],
                angles_rad=PQVA_matrix_prediction[:, PowerFlowFeatureIds.voltage_angle],
                conductances=data.conductances_pu,
                susceptances=data.susceptances_pu
            ),

            denominator=PQVA_matrix_prediction[:, PowerFlowFeatureIds.reactive_power]
        )
        self.relative_reactive_power_errors.append(np.abs(rel_errors))

    def compute(self) -> ndarray:
        return np.array(self.relative_reactive_power_errors)

    @property
    def unit(self) -> str:
        return "ratio"