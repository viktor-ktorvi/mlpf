import random

import numpy as np
import pandapower as pp
import pandapower.networks as pn
import pandas as pd

from pypower.ppoption import ppoption
from pypower.runpf import runpf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from mlpf.data.data.numpy.power_flow import power_flow_data
from mlpf.data.generate.generate_uniform_data import generate_uniform_ppcs
from mlpf.loss.numpy.metrics.active import ActivePowerError, RelativeActivePowerError
from mlpf.loss.numpy.metrics.metrics import MultipleMetrics
from mlpf.loss.numpy.metrics.reactive import ReactivePowerError, RelativeReactivePowerError


def main():
    # Random seeds
    np.random.seed(123)
    random.seed(123)

    # Generate ppcs

    net = pn.case118()
    ppc = pp.converter.to_ppc(net, init="flat")

    base_ppc, converged = runpf(ppc, ppopt=ppoption(OUT_ALL=0, VERBOSE=0))

    ppc_list = generate_uniform_ppcs(
        base_ppc,
        how_many=1000,
        low=0.9,
        high=1.1
    )

    # ppc -> Data
    pf_data_list = []
    for ppc in tqdm(ppc_list, ascii=True, desc="Converting ppcs to data"):
        pf_data_list.append(power_flow_data(ppc))

    data_train, data_val = train_test_split(pf_data_list, test_size=0.33, random_state=42)

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
        RelativeReactivePowerError()
    )

    for i in tqdm(range(predictions_val.shape[0]), ascii=True, desc="Calculating metrics"):
        power_metrics.update(predictions_val[i], data_val[i])

    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 1000)

    pd.set_option('display.float_format', lambda x: "{:2.5f} ".format(x))
    print(power_metrics.compute().describe())


if __name__ == "__main__":
    main()
