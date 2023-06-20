import random

import numpy as np
import pandapower as pp
import pandapower.networks as pn
import pandas as pd

from pandas import DataFrame
from pypower.ppoption import ppoption
from pypower.runpf import runpf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from mlpf.data.data.numpy.power_flow import power_flow_data, get_relative_power_flow_errors
from mlpf.data.generate.generate_uniform_data import generate_uniform_ppcs


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

    val_predictions = model.predict(features_val)

    # power flow evaluation
    relative_active_power_errors_list = []
    relative_reactive_power_errors_list = []

    for i in range(val_predictions.shape[0]):
        relative_active_power_errors, relative_reactive_power_errors = get_relative_power_flow_errors(val_predictions[i], data_val[i])

        relative_active_power_errors_list.append(relative_active_power_errors)
        relative_reactive_power_errors_list.append(relative_reactive_power_errors)

    rel_errors_df = DataFrame(
        data=np.vstack(
            (np.array(relative_active_power_errors_list).flatten(),
             np.array(relative_reactive_power_errors_list).flatten())).T,
        columns=["relative active power error", "relative reactive power error"]
    )

    pd.set_option('display.float_format', lambda x: "{:2.4f} ".format(x))
    print(rel_errors_df.describe())


if __name__ == "__main__":
    main()
