import abc

import pandas as pd

from numpy import ndarray
from pandas import DataFrame
from typing import Union

from mlpf.data.data.numpy.optimal_power_flow import OptimalPowerFlowData
from mlpf.data.data.numpy.power_flow import PowerFlowData


class BaseMetric(abc.ABC):
    """
    A base class for (optimal) power flow metrics for numpy/sklearn.
    """

    @abc.abstractmethod
    def update(self, predictions: ndarray, data: Union[PowerFlowData, OptimalPowerFlowData]):
        """
        Add new prediction sample to the metric.

        :param predictions: (O)PF variable predictions.
        :param data: Corresponding (O)PF data object.
        :return:
        """
        pass

    @abc.abstractmethod
    def compute(self) -> ndarray:
        """
        Return the metric variable array.
        :return:
        """
        pass

    @property
    @abc.abstractmethod
    def unit(self) -> str:
        """
        Unit string to be used for display.
        :return:
        """
        pass


class MultipleMetrics:
    """
    A class to collect and compute multiple BaseMetrics.
    """

    def __init__(self, *metrics: BaseMetric):
        self.metrics = metrics

    def update(self, predictions: ndarray, data: Union[PowerFlowData, OptimalPowerFlowData]):
        for metric in self.metrics:
            metric.update(predictions, data)

    def compute(self) -> DataFrame:
        results = DataFrame()

        for metric in self.metrics:
            metric_df = DataFrame({metric.__class__.__name__ + f" [{metric.unit}]": metric.compute().flatten()})
            results = pd.concat([results, metric_df], axis=1)

        return results
