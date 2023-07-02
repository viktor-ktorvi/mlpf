import random

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from mlpf.data.data.optimal_power_flow import optimal_power_flow_data
from mlpf.data.loading.load_data import load_data
from mlpf.loss.numpy.metrics.active import ActivePowerError, RelativeActivePowerError
from mlpf.loss.numpy.metrics.bounds.active import UpperActivePowerError, LowerActivePowerError
from mlpf.loss.numpy.metrics.bounds.reactive import UpperReactivePowerError, LowerReactivePowerError
from mlpf.loss.numpy.metrics.bounds.voltage import UpperVoltageError, LowerVoltageError
from mlpf.loss.numpy.metrics.costs import ActivePowerCost, RelativeActivePowerCost
from mlpf.loss.numpy.metrics.metrics import MultipleMetrics
from mlpf.loss.numpy.metrics.reactive import ReactivePowerError, RelativeReactivePowerError


def main():
    # Random seeds
    np.random.seed(123)
    random.seed(123)

    ppc_list = load_data("solved_opf_ppcs_case118_10k", max_samples=10000)

    # ppc -> Data
    opf_data_list = []
    for ppc in tqdm(ppc_list, ascii=True, desc="Converting ppcs to data"):
        opf_data_list.append(optimal_power_flow_data(ppc))

    data_train, data_val = train_test_split(opf_data_list, test_size=0.33, random_state=42)

    features_train = np.vstack([data.feature_vector for data in data_train])
    targets_train = np.vstack([data.target_vector for data in data_train])

    features_val = np.vstack([data.feature_vector for data in data_val])
    targets_val = np.vstack([data.target_vector for data in data_val])

    # Model

    backbone = LinearRegression()
    # backbone = MultiOutputRegressor(HuberRegressor())
    # backbone = MultiOutputRegressor(BayesianRidge())

    model = make_pipeline(StandardScaler(), backbone)
    model.fit(features_train, targets_train)

    # Evaluation

    print(f"Train R2 score = {model.score(features_train, targets_train)}")
    print(f"Val R2 score = {model.score(features_val, targets_val)}\n")

    predictions_val = model.predict(features_val)

    power_metrics = MultipleMetrics(
        ActivePowerError(),
        ReactivePowerError(),
        RelativeActivePowerError(),
        RelativeReactivePowerError(),
        ActivePowerCost(),
        RelativeActivePowerCost(),
        UpperVoltageError(),
        LowerVoltageError(),
        UpperActivePowerError(),
        LowerActivePowerError(),
        UpperReactivePowerError(),
        LowerReactivePowerError()
    )

    for i in tqdm(range(predictions_val.shape[0]), ascii=True, desc="Calculating metrics"):
        power_metrics.update(predictions_val[i], data_val[i])

    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 1000)

    pd.set_option('display.float_format', lambda x: "{:2.5f} ".format(x))
    print(power_metrics.compute().describe())


if __name__ == "__main__":
    main()
